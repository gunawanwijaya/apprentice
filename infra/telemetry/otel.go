package telemetry

import (
	"context"
	"errors"
	"log/slog"
	"math"
	"net/http"
	"os"
	"time"

	"go.opentelemetry.io/contrib/bridges/otelslog"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlplog/otlploghttp"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/log/global"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/propagation"
	sdklog "go.opentelemetry.io/otel/sdk/log"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

var httpLatencyBuckets = []float64{
	0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005,
	0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
}

func Init(ctx context.Context, serviceName string) (func(context.Context) error, error) {
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		endpoint = "http://localhost:4318"
	}

	res, err := resource.New(ctx,
		resource.WithAttributes(attribute.String("service.name", serviceName)),
	)
	if err != nil {
		return nil, err
	}

	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	traceExp, err := otlptracehttp.New(ctx, otlptracehttp.WithEndpointURL(endpoint+"/v1/traces"))
	if err != nil {
		return nil, err
	}
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(traceExp),
		sdktrace.WithResource(res),
	)
	otel.SetTracerProvider(tp)

	metricExp, err := otlpmetrichttp.New(ctx, otlpmetrichttp.WithEndpointURL(endpoint+"/v1/metrics"))
	if err != nil {
		return nil, err
	}
	mp := sdkmetric.NewMeterProvider(
		sdkmetric.WithReader(sdkmetric.NewPeriodicReader(metricExp, sdkmetric.WithInterval(5*time.Second))),
		sdkmetric.WithResource(res),
	)
	otel.SetMeterProvider(mp)

	logExp, err := otlploghttp.New(ctx, otlploghttp.WithEndpointURL(endpoint+"/v1/logs"))
	if err != nil {
		return nil, err
	}
	lp := sdklog.NewLoggerProvider(
		sdklog.WithProcessor(sdklog.NewBatchProcessor(logExp)),
		sdklog.WithResource(res),
	)
	global.SetLoggerProvider(lp)

	jsonStdout := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo})
	otelLoki := otelslog.NewHandler(serviceName, otelslog.WithLoggerProvider(lp))

	slog.SetDefault(slog.New(&correlatedHandler{
		service:  serviceName,
		handlers: []slog.Handler{jsonStdout, otelLoki},
	}))

	return func(shutdownCtx context.Context) error {
		return errors.Join(
			tp.Shutdown(shutdownCtx),
			mp.Shutdown(shutdownCtx),
			lp.Shutdown(shutdownCtx),
		)
	}, nil
}

// Middleware extracts W3C trace headers, starts a server span, records latency with trace exemplars,
// and emits a trace-correlated JSON/OTLP access log.
func Middleware(serviceName string, next http.Handler) http.Handler {
	tracer := otel.Tracer(serviceName + "/http")
	meter := otel.Meter(serviceName)
	prop := otel.GetTextMapPropagator()
	reqDur, _ := meter.Float64Histogram(
		"http.server.request.duration",
		metric.WithUnit("s"),
		metric.WithDescription("HTTP server request duration in seconds"),
		metric.WithExplicitBucketBoundaries(httpLatencyBuckets...),
	)

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		route := r.URL.Path

		parentCtx := prop.Extract(r.Context(), propagation.HeaderCarrier(r.Header))
		rCtx, span := tracer.Start(parentCtx, r.Method+" "+route, trace.WithSpanKind(trace.SpanKindServer))
		defer span.End()

		if sc := span.SpanContext(); sc.IsValid() {
			w.Header().Set("X-Trace-Id", sc.TraceID().String())
		}

		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r.WithContext(rCtx))

		elapsed := time.Since(start)
		durSec := elapsed.Seconds()
		latencyMS := round6(float64(elapsed.Nanoseconds()) / 1e6)

		span.SetAttributes(
			attribute.String("http.route", route),
			attribute.String("http.request.method", r.Method),
			attribute.Int("http.response.status_code", rec.status),
		)
		// Passing rCtx attaches the active TraceID as a Prometheus Exemplar on this histogram bucket.
		reqDur.Record(rCtx, durSec, metric.WithAttributes(attribute.String("http.route", route)))

		slog.InfoContext(rCtx, "http_request",
			"method", r.Method,
			"route", route,
			"status", rec.status,
			"latency_ms", latencyMS,
		)
	})
}

func round6(v float64) float64 {
	return math.Round(v*1e6) / 1e6
}

func sanitizeAttr(a slog.Attr) slog.Attr {
	switch a.Value.Kind() {
	case slog.KindFloat64:
		return slog.Float64(a.Key, round6(a.Value.Float64()))
	case slog.KindGroup:
		orig := a.Value.Group()
		cleaned := make([]any, len(orig))
		for i, g := range orig {
			cleaned[i] = sanitizeAttr(g)
		}
		return slog.Group(a.Key, cleaned...)
	default:
		return a
	}
}

type correlatedHandler struct {
	service  string
	handlers []slog.Handler
}

func (c *correlatedHandler) Enabled(ctx context.Context, level slog.Level) bool {
	for _, h := range c.handlers {
		if h.Enabled(ctx, level) {
			return true
		}
	}
	return false
}

func (c *correlatedHandler) Handle(ctx context.Context, r slog.Record) error {
	clean := slog.NewRecord(r.Time, r.Level, r.Message, r.PC)
	clean.AddAttrs(slog.String("service", c.service))

	if sc := trace.SpanContextFromContext(ctx); sc.IsValid() {
		clean.AddAttrs(
			slog.String("trace_id", sc.TraceID().String()),
			slog.String("span_id", sc.SpanID().String()),
		)
	}
	r.Attrs(func(a slog.Attr) bool {
		clean.AddAttrs(sanitizeAttr(a))
		return true
	})

	var errs []error
	for _, h := range c.handlers {
		if h.Enabled(ctx, clean.Level) {
			errs = append(errs, h.Handle(ctx, clean.Clone()))
		}
	}
	return errors.Join(errs...)
}

func (c *correlatedHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	sanitized := make([]slog.Attr, len(attrs))
	for i, a := range attrs {
		sanitized[i] = sanitizeAttr(a)
	}
	next := make([]slog.Handler, len(c.handlers))
	for i, h := range c.handlers {
		next[i] = h.WithAttrs(sanitized)
	}
	return &correlatedHandler{service: c.service, handlers: next}
}

func (c *correlatedHandler) WithGroup(name string) slog.Handler {
	next := make([]slog.Handler, len(c.handlers))
	for i, h := range c.handlers {
		next[i] = h.WithGroup(name)
	}
	return &correlatedHandler{service: c.service, handlers: next}
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (sr *statusRecorder) WriteHeader(code int) {
	sr.status = code
	sr.ResponseWriter.WriteHeader(code)
}
