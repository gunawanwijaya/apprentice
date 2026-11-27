package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gunawanwijaya/apprentice/07-agents-mcp-evals/go/internal/contract"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/propagation"
)

type ctxKey struct{}

// Principal is bound strictly from verified session/JWT headers—never from LLM tool arguments.
type Principal struct {
	TenantID      string
	UserID        string
	AllowMutation bool
}

func WithPrincipal(ctx context.Context, p Principal) context.Context {
	return context.WithValue(ctx, ctxKey{}, p)
}

func PrincipalFromContext(ctx context.Context) (Principal, bool) {
	p, ok := ctx.Value(ctxKey{}).(Principal)
	return p, ok
}

type Order struct {
	OrderID       string  `json:"order_id"`
	TenantID      string  `json:"tenant_id"`
	UserID        string  `json:"user_id"`
	Status        string  `json:"status"`
	PaidAmountIDR int64   `json:"paid_amount_idr"`
	SLABreachProb float64 `json:"sla_breach_prob"`
	ETAP50Hours   int     `json:"eta_p50_hours"`
	ETAP90Hours   int     `json:"eta_p90_hours"`
	Refundable    bool    `json:"refundable"`
	SellerNote    string  `json:"seller_note"`
}

type ToolSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	ReadOnly    bool           `json:"readOnlyHint"`
	InputSchema map[string]any `json:"inputSchema"`
}

type ToolResult struct {
	Code    string         `json:"code"`
	Data    map[string]any `json:"data"`
	IsError bool           `json:"is_error"`
}

type Server struct {
	orders        map[string]Order
	policyMap     map[string]string
	ragServiceURL string
	httpClient    *http.Client
	toolCalls     metric.Int64Counter
}

var tracer = otel.Tracer("07-agents-mcp-evals/mcp")

func NewServer(orders []Order, policyMap map[string]string, ragServiceURL string) *Server {
	byID := make(map[string]Order, len(orders))
	for _, o := range orders {
		byID[strings.ToUpper(o.OrderID)] = o
	}
	pm := map[string]string{
		"ORDER":    "POL-ORDE-01#0",
		"REFUND":   "POL-REFU-01#0",
		"SHIPPING": "POL-SHIP-01#0",
		"ACCOUNT":  "POL-ACCO-01#0",
	}
	for k, v := range policyMap {
		pm[k] = v
	}
	meter := otel.Meter("07-agents-mcp-evals/mcp")
	tc, _ := meter.Int64Counter(
		"mcp.tool.calls",
		metric.WithDescription("Total MCP tool invocations partitioned by tool and outcome"),
	)
	return &Server{
		orders:        byID,
		policyMap:     pm,
		ragServiceURL: strings.TrimRight(ragServiceURL, "/"),
		httpClient:    &http.Client{Timeout: 3 * time.Second},
		toolCalls:     tc,
	}
}

func LoadOrdersSeed(path string) ([]Order, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var orders []Order
	if err := json.Unmarshal(raw, &orders); err != nil {
		return nil, err
	}
	return orders, nil
}

func LoadPolicyFallbackMap(ragCorpusJSONL string) map[string]string {
	out := make(map[string]string)
	raw, err := os.ReadFile(ragCorpusJSONL)
	if err != nil {
		return out
	}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var row struct {
			TenantID string `json:"tenant_id"`
			ChunkID  string `json:"chunk_id"`
		}
		if json.Unmarshal([]byte(line), &row) == nil && row.TenantID != "" {
			if _, exists := out[row.TenantID]; !exists {
				out[row.TenantID] = row.ChunkID
			}
		}
	}
	return out
}

func (s *Server) OrdersCount() int {
	return len(s.orders)
}

func (s *Server) OrderByID(orderID string) (Order, bool) {
	o, ok := s.orders[strings.ToUpper(strings.TrimSpace(orderID))]
	return o, ok
}

// ToolSchemas intentionally omits tenant_id and user_id to prevent Confused Deputy escalation.
func (s *Server) ToolSchemas() []ToolSchema {
	return []ToolSchema{
		{
			Name:        "get_order_dossier",
			Description: "Fetch order status, payment amount, and courier note within the caller's verified tenant.",
			ReadOnly:    true,
			InputSchema: map[string]any{
				"type":     "object",
				"required": []string{"order_id"},
				"properties": map[string]any{
					"order_id": map[string]any{"type": "string", "pattern": "^ORD-[0-9]{4}$"},
				},
			},
		},
		{
			Name:        "score_sla_risk",
			Description: "Retrieve LightGBM SLA breach probability and CQR p50/p90 ETA hours (Module 01 contract).",
			ReadOnly:    true,
			InputSchema: map[string]any{
				"type":     "object",
				"required": []string{"order_id"},
				"properties": map[string]any{
					"order_id": map[string]any{"type": "string"},
				},
			},
		},
		{
			Name:        "search_tenant_policy",
			Description: "Query Hybrid BM25 + Vector RRF policy chunks for the caller's tenant (Module 06 contract).",
			ReadOnly:    true,
			InputSchema: map[string]any{
				"type":     "object",
				"required": []string{"query"},
				"properties": map[string]any{
					"query": map[string]any{"type": "string"},
				},
			},
		},
		{
			Name:        "submit_refund_action",
			Description: "Mutating tool: submit a policy-gated refund or generate a Human-in-the-Loop approval draft.",
			ReadOnly:    false,
			InputSchema: map[string]any{
				"type":     "object",
				"required": []string{"order_id", "amount_idr"},
				"properties": map[string]any{
					"order_id":   map[string]any{"type": "string"},
					"amount_idr": map[string]any{"type": "integer", "minimum": 1},
				},
			},
		},
	}
}

func (s *Server) CallTool(ctx context.Context, name string, args map[string]any) (ToolResult, error) {
	ctx, span := tracer.Start(ctx, "mcp.tool."+name)
	defer span.End()

	p, ok := PrincipalFromContext(ctx)
	if !ok || strings.TrimSpace(p.TenantID) == "" {
		s.recordMetric(ctx, name, "unauthorized")
		return ToolResult{Code: "unauthorized", IsError: true}, errors.New("missing verified session principal")
	}
	tenant := strings.ToUpper(strings.TrimSpace(p.TenantID))
	span.SetAttributes(attribute.String("mcp.tool", name), attribute.String("tenant_id", tenant))

	switch name {
	case "get_order_dossier":
		orderID, _ := args["order_id"].(string)
		o, exists := s.OrderByID(orderID)
		if !exists {
			s.recordMetric(ctx, name, "not_found")
			return ToolResult{Code: "not_found", IsError: true, Data: map[string]any{"error": "order not found"}}, nil
		}
		if o.TenantID != tenant {
			s.recordMetric(ctx, name, "forbidden_cross_tenant")
			return ToolResult{
				Code:    "forbidden_cross_tenant",
				IsError: true,
				Data:    map[string]any{"error": "order belongs to another tenant"},
			}, nil
		}
		s.recordMetric(ctx, name, "ok")
		return ToolResult{
			Code: "ok",
			Data: map[string]any{
				"order_id":        o.OrderID,
				"status":          o.Status,
				"paid_amount_idr": o.PaidAmountIDR,
				"refundable":      o.Refundable,
				"untrusted_note":  "<untrusted_seller_note>" + o.SellerNote + "</untrusted_seller_note>",
			},
		}, nil

	case "score_sla_risk":
		orderID, _ := args["order_id"].(string)
		o, exists := s.OrderByID(orderID)
		if !exists || o.TenantID != tenant {
			s.recordMetric(ctx, name, "error")
			return ToolResult{Code: "forbidden_or_missing", IsError: true}, nil
		}
		s.recordMetric(ctx, name, "ok")
		return ToolResult{
			Code: "ok",
			Data: map[string]any{
				"order_id":        o.OrderID,
				"sla_breach_prob": o.SLABreachProb,
				"eta_p50_hours":   o.ETAP50Hours,
				"eta_p90_hours":   o.ETAP90Hours,
			},
		}, nil

	case "search_tenant_policy":
		q, _ := args["query"].(string)
		citation := s.lookupPolicyCitation(ctx, tenant, q)
		s.recordMetric(ctx, name, "ok")
		return ToolResult{
			Code: "ok",
			Data: map[string]any{
				"tenant_id": tenant,
				"citation":  citation,
			},
		}, nil

	case "submit_refund_action":
		if contract.LockMutatingToolsOnReadonlyIntent && !p.AllowMutation {
			s.recordMetric(ctx, name, "blocked_readonly_intent")
			return ToolResult{
				Code:    "mutating_blocked_readonly_intent",
				IsError: true,
				Data:    map[string]any{"error": "mutating tool locked on read-only user intent"},
			}, nil
		}
		orderID, _ := args["order_id"].(string)
		o, exists := s.OrderByID(orderID)
		if !exists || o.TenantID != tenant {
			s.recordMetric(ctx, name, "forbidden_cross_tenant")
			return ToolResult{Code: "forbidden_cross_tenant", IsError: true}, nil
		}
		amt := toInt64(args["amount_idr"])
		if !o.Refundable {
			s.recordMetric(ctx, name, "rejected_ineligible")
			return ToolResult{
				Code:    "rejected_ineligible_order",
				IsError: true,
				Data:    map[string]any{"error": "order status is not eligible for refund"},
			}, nil
		}
		if amt > o.PaidAmountIDR {
			s.recordMetric(ctx, name, "rejected_exceeds_paid")
			return ToolResult{
				Code:    "rejected_exceeds_paid_amount",
				IsError: true,
				Data:    map[string]any{"error": "requested refund exceeds paid_amount_idr"},
			}, nil
		}
		if amt > contract.MaxAutoRefundIDR {
			s.recordMetric(ctx, name, "hitl_draft")
			return ToolResult{
				Code: "hitl_approval_draft",
				Data: map[string]any{
					"draft_id":   "DRF-" + o.OrderID,
					"amount_idr": amt,
					"reason":     fmt.Sprintf("exceeds auto-refund cap (%d IDR)", contract.MaxAutoRefundIDR),
				},
			}, nil
		}
		s.recordMetric(ctx, name, "auto_approved")
		return ToolResult{
			Code: "auto_refund_approved",
			Data: map[string]any{
				"refund_id":  "RFN-" + o.OrderID,
				"amount_idr": amt,
			},
		}, nil

	default:
		s.recordMetric(ctx, name, "unknown_tool")
		return ToolResult{Code: "unknown_tool", IsError: true}, fmt.Errorf("unknown tool %q", name)
	}
}

func (s *Server) lookupPolicyCitation(ctx context.Context, tenant, query string) string {
	if s.ragServiceURL != "" {
		payload, _ := json.Marshal(map[string]string{"tenant_id": tenant, "query": query})
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.ragServiceURL+"/v1/rag/query", bytes.NewReader(payload))
		if err == nil {
			req.Header.Set("Content-Type", "application/json")
			otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(req.Header))
			if resp, err := s.httpClient.Do(req); err == nil {
				defer resp.Body.Close()
				var out struct {
					Citations []string `json:"citations"`
				}
				if json.NewDecoder(resp.Body).Decode(&out) == nil && len(out.Citations) > 0 {
					return out.Citations[0]
				}
			}
		}
	}
	if c, ok := s.policyMap[tenant]; ok {
		return c
	}
	return "POL-" + tenant + "-01#0"
}

func (s *Server) recordMetric(ctx context.Context, tool, outcome string) {
	s.toolCalls.Add(ctx, 1, metric.WithAttributes(
		attribute.String("tool", tool),
		attribute.String("outcome", outcome),
	))
}

// JSON-RPC 2.0 Streamable HTTP Handler (POST /mcp)
type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string    `json:"jsonrpc"`
	ID      any       `json:"id"`
	Result  any       `json:"result,omitempty"`
	Error   *rpcError `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var req rpcRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeRPC(w, http.StatusBadRequest, rpcResponse{JSONRPC: "2.0", Error: &rpcError{Code: -32700, Message: "parse error"}})
		return
	}

	ctx := WithPrincipal(r.Context(), Principal{
		TenantID:      r.Header.Get("X-Tenant-Id"),
		UserID:        r.Header.Get("X-User-Id"),
		AllowMutation: strings.EqualFold(r.Header.Get("X-Allow-Mutation"), "true"),
	})

	switch req.Method {
	case "initialize":
		writeRPC(w, http.StatusOK, rpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result: map[string]any{
				"protocolVersion": contract.MCPProtocolVersion,
				"serverInfo":      map[string]string{"name": "07-agents-mcp-evals", "version": contract.SchemaVersion},
				"capabilities":    map[string]any{"tools": map[string]any{}, "resources": map[string]any{}},
			},
		})
	case "tools/list":
		writeRPC(w, http.StatusOK, rpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result:  map[string]any{"tools": s.ToolSchemas()},
		})
	case "tools/call":
		var p struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		if err := json.Unmarshal(req.Params, &p); err != nil {
			writeRPC(w, http.StatusBadRequest, rpcResponse{JSONRPC: "2.0", ID: req.ID, Error: &rpcError{Code: -32602, Message: err.Error()}})
			return
		}
		res, err := s.CallTool(ctx, p.Name, p.Arguments)
		if err != nil && res.Code == "unauthorized" {
			writeRPC(w, http.StatusForbidden, rpcResponse{JSONRPC: "2.0", ID: req.ID, Error: &rpcError{Code: -32001, Message: err.Error()}})
			return
		}
		txt, _ := json.Marshal(res)
		writeRPC(w, http.StatusOK, rpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result: map[string]any{
				"isError": res.IsError,
				"content": []map[string]string{{"type": "text", "text": string(txt)}},
			},
		})
	case "resources/list":
		writeRPC(w, http.StatusOK, rpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Result: map[string]any{
				"resources": []map[string]string{
					{"uri": "policy://current", "name": "Tenant Grounded Policy Index", "mimeType": "application/json"},
				},
			},
		})
	default:
		writeRPC(w, http.StatusNotFound, rpcResponse{
			JSONRPC: "2.0",
			ID:      req.ID,
			Error:   &rpcError{Code: -32601, Message: "method not found"},
		})
	}
}

func writeRPC(w http.ResponseWriter, status int, resp rpcResponse) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(resp)
}

func toInt64(v any) int64 {
	switch n := v.(type) {
	case int64:
		return n
	case int:
		return int64(n)
	case float64:
		return int64(n)
	default:
		return 0
	}
}
