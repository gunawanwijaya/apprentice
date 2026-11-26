package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/gunawanwijaya/apprentice/05-llm-gateway/go/internal/contract"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model          string    `json:"model"`
	Messages       []Message `json:"messages"`
	Temperature    float64   `json:"temperature"`
	MaxTokens      int       `json:"max_tokens,omitempty"`
	Stream         bool      `json:"stream,omitempty"`
	ResponseSchema string    `json:"response_schema,omitempty"`
	FaultInject    string    `json:"fault_inject,omitempty"`
}

type ChatResult struct {
	Model            string
	Provider         string
	Content          string
	FinishReason     string
	PromptTokens     int
	CompletionTokens int
	FailedOver       bool
}

type StreamDelta struct {
	Token            string
	Done             bool
	PromptTokens     int
	CompletionTokens int
}

type UpstreamError struct {
	StatusCode int
	Message    string
}

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("upstream HTTP %d: %s", e.StatusCode, e.Message)
}

type Client interface {
	Complete(ctx context.Context, model string, req ChatRequest) (ChatResult, error)
	Stream(ctx context.Context, model string, req ChatRequest, onDelta func(StreamDelta) error) error
}

type Router struct {
	Primary   Client
	Secondary Client
}

func (r *Router) CompleteWithFailover(ctx context.Context, req ChatRequest) (ChatResult, error) {
	primaryModel := req.Model
	if primaryModel == "" {
		primaryModel = contract.PrimaryModel
	}

	var err error
	var res ChatResult
	if req.FaultInject == "force_live_failover" {
		err = &UpstreamError{StatusCode: http.StatusTooManyRequests, Message: "simulated primary 429"}
	} else {
		res, err = r.Primary.Complete(ctx, primaryModel, req)
		if err == nil {
			return res, nil
		}
	}

	var upErr *UpstreamError
	if errors.As(err, &upErr) && contract.RetryableStatusCodes[upErr.StatusCode] && r.Secondary != nil {
		fbRes, fbErr := r.Secondary.Complete(ctx, contract.FallbackModel, req)
		if fbErr == nil {
			fbRes.FailedOver = true
			return fbRes, nil
		}
		return ChatResult{}, fbErr
	}
	return ChatResult{}, err
}

type OpenAIHTTPClient struct {
	ProviderName  string
	BaseURL       string
	APIKey        string
	ModelOverride string
	HTTPClient    *http.Client
}

func chatCompletionsURL(base string) string {
	b := strings.TrimRight(base, "/")
	switch {
	case strings.HasSuffix(b, "/chat/completions"):
		return b
	case strings.HasSuffix(b, "/v1"), strings.HasSuffix(b, "/openai"):
		return b + "/chat/completions"
	default:
		return b + "/v1/chat/completions"
	}
}

func (c *OpenAIHTTPClient) resolveModel(requested string) string {
	if c.ModelOverride != "" {
		return c.ModelOverride
	}
	return requested
}

func (c *OpenAIHTTPClient) Complete(ctx context.Context, model string, req ChatRequest) (ChatResult, error) {
	wireModel := c.resolveModel(model)
	payload := map[string]any{
		"model":       wireModel,
		"messages":    req.Messages,
		"temperature": req.Temperature,
		"max_tokens":  req.MaxTokens,
		"stream":      false,
	}
	if req.ResponseSchema != "" {
		payload["response_format"] = map[string]any{"type": "json_object"}
	}
	body, _ := json.Marshal(payload)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, chatCompletionsURL(c.BaseURL), bytes.NewReader(body))
	if err != nil {
		return ChatResult{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	if req.FaultInject != "" {
		httpReq.Header.Set("X-Mock-Fault", req.FaultInject)
	}
	if req.ResponseSchema != "" {
		httpReq.Header.Set("X-Mock-Schema", req.ResponseSchema)
	}

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return ChatResult{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return ChatResult{}, &UpstreamError{StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(raw))}
	}

	var parsed struct {
		Model   string `json:"model"`
		Choices []struct {
			Message      Message `json:"message"`
			FinishReason string  `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return ChatResult{}, err
	}
	if len(parsed.Choices) == 0 {
		return ChatResult{}, errors.New("empty upstream choices")
	}
	outModel := parsed.Model
	if outModel == "" {
		outModel = wireModel
	}
	return ChatResult{
		Model:            outModel,
		Provider:         c.ProviderName,
		Content:          parsed.Choices[0].Message.Content,
		FinishReason:     parsed.Choices[0].FinishReason,
		PromptTokens:     parsed.Usage.PromptTokens,
		CompletionTokens: parsed.Usage.CompletionTokens,
	}, nil
}

func (c *OpenAIHTTPClient) Stream(ctx context.Context, model string, req ChatRequest, onDelta func(StreamDelta) error) error {
	wireModel := c.resolveModel(model)
	payload := map[string]any{
		"model":          wireModel,
		"messages":       req.Messages,
		"temperature":    req.Temperature,
		"max_tokens":     req.MaxTokens,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	}
	body, _ := json.Marshal(payload)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, chatCompletionsURL(c.BaseURL), bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	if req.FaultInject != "" {
		httpReq.Header.Set("X-Mock-Fault", req.FaultInject)
	}

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return &UpstreamError{StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(raw))}
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 512*1024)
	sawFinish := false

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			return onDelta(StreamDelta{Done: true})
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage,omitempty"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("corrupt upstream SSE chunk: %w", err)
		}
		delta := StreamDelta{}
		if len(chunk.Choices) > 0 {
			delta.Token = chunk.Choices[0].Delta.Content
			if chunk.Choices[0].FinishReason != "" {
				sawFinish = true
			}
		}
		if chunk.Usage != nil {
			delta.PromptTokens = chunk.Usage.PromptTokens
			delta.CompletionTokens = chunk.Usage.CompletionTokens
		}
		if err := onDelta(delta); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	if sawFinish {
		return onDelta(StreamDelta{Done: true})
	}
	return errors.New("upstream stream closed before [DONE]")
}

type MockUpstreamStats struct {
	ActiveStreams  atomic.Int64
	AbortedStreams atomic.Int64
	GeneratedToks  atomic.Int64
	badJSONCounter atomic.Int64
}

func NewMockUpstreamHandler(providerName string, prefillDelay, perTokenDelay time.Duration, stats *MockUpstreamStats) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid json", http.StatusBadRequest)
			return
		}
		if req.FaultInject == "" {
			req.FaultInject = r.Header.Get("X-Mock-Fault")
		}
		if req.ResponseSchema == "" {
			req.ResponseSchema = r.Header.Get("X-Mock-Schema")
		}

		var joined strings.Builder
		for _, m := range req.Messages {
			joined.WriteString(m.Content)
			joined.WriteByte(' ')
		}
		promptText := joined.String()
		if len(promptText) > 32000 {
			http.Error(w, "context_length_exceeded", http.StatusBadRequest)
			return
		}
		promptTokens := max(8, len(promptText)/4)

		if req.FaultInject == "primary_429" && providerName == "openai" {
			http.Error(w, `{"error":"rate_limit_exceeded"}`, http.StatusTooManyRequests)
			return
		}

		select {
		case <-time.After(prefillDelay):
		case <-r.Context().Done():
			return
		}

		reply := buildDeterministicReply(promptText, req)
		if req.FaultInject == "bad_json_once" && stats.badJSONCounter.Add(1) == 1 {
			reply = "```json\n{\"verdict\": \"APPROVE\", BROKEN_JSON"
		}

		words := strings.Fields(reply)
		if req.MaxTokens > 0 && len(words) > req.MaxTokens {
			words = words[:req.MaxTokens]
		}

		if !req.Stream {
			decodeWait := time.Duration(len(words)) * perTokenDelay
			select {
			case <-time.After(decodeWait):
			case <-r.Context().Done():
				return
			}
			finish := "stop"
			if req.MaxTokens > 0 && len(words) == req.MaxTokens && req.MaxTokens < 6 {
				finish = "length"
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model": req.Model,
				"choices": []map[string]any{{
					"message":       Message{Role: "assistant", Content: strings.Join(words, " ")},
					"finish_reason": finish,
				}},
				"usage": map[string]int{
					"prompt_tokens":     promptTokens,
					"completion_tokens": len(words),
				},
			})
			return
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		stats.ActiveStreams.Add(1)
		defer stats.ActiveStreams.Add(-1)

		for i, tok := range words {
			if req.FaultInject == "midstream_cut" && i == 3 {
				return
			}
			select {
			case <-r.Context().Done():
				stats.AbortedStreams.Add(1)
				return
			case <-time.After(perTokenDelay):
				piece := tok
				if i > 0 {
					piece = " " + tok
				}
				stats.GeneratedToks.Add(1)
				chunk, _ := json.Marshal(map[string]any{
					"choices": []map[string]any{{"delta": map[string]string{"content": piece}}},
				})
				_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
				flusher.Flush()
			}
		}

		usageChunk, _ := json.Marshal(map[string]any{
			"choices": []any{},
			"usage": map[string]int{
				"prompt_tokens":     promptTokens,
				"completion_tokens": len(words),
			},
		})
		_, _ = fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", usageChunk)
		flusher.Flush()
	})
	return mux
}

func buildDeterministicReply(prompt string, req ChatRequest) string {
	low := strings.ToLower(prompt)
	if req.ResponseSchema == "refund_decision" {
		if strings.Contains(low, "10042") {
			return `{"order_id":"10042","verdict":"APPROVE","approved_idr":150000,"reason_code":"DAMAGED_PARCEL"}`
		}
		if strings.Contains(low, "10049") {
			return `{"order_id":"10049","verdict":"DENY","approved_idr":0,"reason_code":"OUTSIDE_SLA_WINDOW"}`
		}
		return `{"order_id":"UNKNOWN","verdict":"MANUAL_REVIEW","approved_idr":0,"reason_code":"INSUFFICIENT_CONTEXT"}`
	}
	if strings.Contains(low, "reset") && strings.Contains(low, "pin") {
		return "Open Settings > Security > Reset PIN, verify via SMS OTP, and choose a new 6-digit PIN."
	}
	if req.MaxTokens >= 120 {
		return strings.Repeat("Detailed policy clause analysis and verbose justification text for audit trail. ", 12)
	}
	return " Streamed response tokens generated cleanly through the Go LLM control plane with context cancellation armed."
}
