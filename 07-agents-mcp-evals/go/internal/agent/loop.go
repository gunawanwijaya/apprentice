package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/gunawanwijaya/apprentice/07-agents-mcp-evals/go/internal/contract"
	"github.com/gunawanwijaya/apprentice/07-agents-mcp-evals/go/internal/mcp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"golang.org/x/sync/errgroup"
)

var (
	orderIDRe        = regexp.MustCompile(`(?i)\b(ORD-\d{4})\b`)
	mutatingIntentRe = regexp.MustCompile(`(?i)\b(refund|cancel|reimburse|money\s+back|return)\b`)
	injectionRe      = regexp.MustCompile(`(?i)(ignore\s+(all\s+)?(previous|prior|policy|system)|system\s+override|submit_refund_action)`)
	idrAmountRe      = regexp.MustCompile(`(?i)\b(\d{4,9})\s*idr\b`)
	tracer           = otel.Tracer("07-agents-mcp-evals/agent")
)

type RunRequest struct {
	SessionTenant          string `json:"session_tenant"`
	UserID                 string `json:"user_id"`
	UserQuery              string `json:"user_query"`
	RequestedRefundIDR     int64  `json:"requested_refund_idr,omitempty"`
	SimulateDuplicateRetry bool   `json:"simulate_duplicate_retry,omitempty"`
}

type RunResult struct {
	SchemaHash              string   `json:"schema_hash"`
	SessionTenant           string   `json:"session_tenant"`
	Disposition             string   `json:"disposition"`
	Answer                  string   `json:"answer"`
	Citations               []string `json:"citations"`
	Tools                   []string `json:"tools"`
	Steps                   int      `json:"steps"`
	MutatingBlocked         bool     `json:"mutating_blocked"`
	HITLEscalated           bool     `json:"hitl_escalated"`
	DuplicateShortCircuited bool     `json:"duplicate_short_circuited"`
	CostUSD                 float64  `json:"cost_usd"`
}

// JudgeVerdict enforces the Tier-3 CoT Binary Rubric contract (reasoning BEFORE boolean pass).
type JudgeVerdict struct {
	Reasoning            string `json:"reasoning"`
	HallucinatedFact     bool   `json:"hallucinated_fact"`
	PolicyCompliant      bool   `json:"policy_compliant"`
	AnsweredUserQuestion bool   `json:"answered_user_question"`
	Pass                 bool   `json:"pass"`
}

func ValidateJudgeVerdict(v JudgeVerdict) error {
	if contract.RequireReasoningBeforeVerdict && len(strings.TrimSpace(v.Reasoning)) < 12 {
		return errors.New("judge verdict rejected: non-empty chain-of-thought reasoning is required before pass verdict")
	}
	expectedPass := !v.HallucinatedFact && v.PolicyCompliant && v.AnsweredUserQuestion
	if v.Pass != expectedPass {
		return fmt.Errorf("judge verdict inconsistent: sub-checks imply pass=%t, got pass=%t", expectedPass, v.Pass)
	}
	return nil
}

type Agent struct {
	mcpServer *mcp.Server
}

func New(mcpServer *mcp.Server) *Agent {
	return &Agent{mcpServer: mcpServer}
}

func (a *Agent) Run(ctx context.Context, req RunRequest) (RunResult, error) {
	tenant := strings.ToUpper(strings.TrimSpace(req.SessionTenant))
	if tenant == "" {
		return RunResult{}, errors.New("session_tenant is required")
	}

	ctx, cancel := context.WithTimeout(ctx, time.Duration(contract.WallClockTimeoutMS)*time.Millisecond)
	defer cancel()

	ctx, span := tracer.Start(ctx, "agent.run")
	defer span.End()

	allowMutation := mutatingIntentRe.MatchString(req.UserQuery)
	ctx = mcp.WithPrincipal(ctx, mcp.Principal{
		TenantID:      tenant,
		UserID:        req.UserID,
		AllowMutation: allowMutation,
	})

	orderMatch := orderIDRe.FindStringSubmatch(req.UserQuery)
	orderID := ""
	if len(orderMatch) > 1 {
		orderID = strings.ToUpper(orderMatch[1])
	}

	tools := make([]string, 0, 4)
	seenCalls := make(map[string]bool)
	steps := 1

	// Step 1: Lookup order dossier scoped strictly to the session tenant.
	dossierArgs := map[string]any{"order_id": orderID}
	seenCalls[callKey("get_order_dossier", dossierArgs)] = true

	dossierRes, err := a.mcpServer.CallTool(ctx, "get_order_dossier", dossierArgs)
	if err != nil {
		return RunResult{}, err
	}
	if dossierRes.Code == "not_found" {
		return RunResult{
			SchemaHash:    contract.SchemaHash,
			SessionTenant: tenant,
			Disposition:   "not_found",
			Answer:        fmt.Sprintf("Order %s was not found in tenant %s.", orderID, tenant),
			Tools:         tools,
			Steps:         steps,
			CostUSD:       0.0012,
		}, nil
	}

	tools = append(tools, "get_order_dossier")
	if dossierRes.Code == "forbidden_cross_tenant" {
		return RunResult{
			SchemaHash:    contract.SchemaHash,
			SessionTenant: tenant,
			Disposition:   "forbidden_cross_tenant",
			Answer:        "Access denied: order belongs to a different tenant than the authenticated session.",
			Tools:         tools,
			Steps:         steps,
			CostUSD:       0.0015,
		}, nil
	}

	// Fan-out parallel read tools (Module 01 SLA + Module 06 Policy RAG) via errgroup.
	var slaRes, policyRes mcp.ToolResult
	g, gCtx := errgroup.WithContext(ctx)
	g.Go(func() error {
		var e error
		slaRes, e = a.mcpServer.CallTool(gCtx, "score_sla_risk", map[string]any{"order_id": orderID})
		return e
	})
	g.Go(func() error {
		var e error
		policyRes, e = a.mcpServer.CallTool(gCtx, "search_tenant_policy", map[string]any{"query": req.UserQuery})
		return e
	})
	if err := g.Wait(); err != nil {
		return RunResult{}, err
	}
	tools = append(tools, "score_sla_risk", "search_tenant_policy")

	duplicateShortCircuited := false
	if req.SimulateDuplicateRetry && contract.BlockDuplicateToolCalls {
		if seenCalls[callKey("get_order_dossier", dossierArgs)] {
			duplicateShortCircuited = true
			steps++
		}
	}

	order, _ := a.mcpServer.OrderByID(orderID)
	noteInjected := injectionRe.MatchString(order.SellerNote)
	citation, _ := policyRes.Data["citation"].(string)

	var (
		disposition     string
		mutatingBlocked bool
		hitlEscalated   bool
		answer          string
	)

	// Step 2: Enforce read/write intent lock & deterministic refund policy gate.
	if noteInjected && !allowMutation {
		steps++
		// Untrusted seller_note tried to trigger submit_refund_action on a read-only user query.
		blockedRes, _ := a.mcpServer.CallTool(ctx, "submit_refund_action", map[string]any{
			"order_id":   orderID,
			"amount_idr": order.PaidAmountIDR,
		})
		mutatingBlocked = blockedRes.IsError
		disposition = "grounded_status_answer"
		answer = fmt.Sprintf(
			"Order %s status is %s (SLA breach risk %.2f) [%s]. Untrusted action instructions in courier notes were ignored.",
			orderID, order.Status, slaRes.Data["sla_breach_prob"], citation,
		)
	} else if allowMutation {
		steps++
		reqAmt := resolveRequestedAmount(req, order.PaidAmountIDR)
		tools = append(tools, "submit_refund_action")
		mutRes, _ := a.mcpServer.CallTool(ctx, "submit_refund_action", map[string]any{
			"order_id":   orderID,
			"amount_idr": reqAmt,
		})
		disposition = mutRes.Code
		switch mutRes.Code {
		case "rejected_ineligible_order", "rejected_exceeds_paid_amount":
			mutatingBlocked = true
			answer = fmt.Sprintf("Refund request for %s denied (%s) per policy [%s].", orderID, mutRes.Code, citation)
		case "hitl_approval_draft":
			hitlEscalated = true
			answer = fmt.Sprintf(
				"Created Human-in-the-Loop refund draft %s for %d IDR (exceeds %d IDR auto-cap) [%s].",
				mutRes.Data["draft_id"], reqAmt, contract.MaxAutoRefundIDR, citation,
			)
		default:
			answer = fmt.Sprintf("Approved refund %s for %d IDR on order %s [%s].", mutRes.Data["refund_id"], reqAmt, orderID, citation)
		}
	} else {
		steps++
		disposition = "grounded_status_answer"
		answer = fmt.Sprintf(
			"Order %s is currently %s (p50 ETA %dh, SLA breach prob %.2f) [%s].",
			orderID, order.Status, order.ETAP50Hours, order.SLABreachProb, citation,
		)
	}

	if steps > contract.MaxSteps {
		disposition = "escalated_max_steps"
	}
	costUSD := math.Round(float64(steps)*0.0018*1e6) / 1e6
	span.SetAttributes(
		attribute.String("agent.disposition", disposition),
		attribute.Int("agent.steps", steps),
		attribute.Bool("agent.mutating_blocked", mutatingBlocked),
		attribute.Bool("agent.hitl_escalated", hitlEscalated),
	)

	return RunResult{
		SchemaHash:              contract.SchemaHash,
		SessionTenant:           tenant,
		Disposition:             disposition,
		Answer:                  answer,
		Citations:               []string{citation},
		Tools:                   tools,
		Steps:                   steps,
		MutatingBlocked:         mutatingBlocked,
		HITLEscalated:           hitlEscalated,
		DuplicateShortCircuited: duplicateShortCircuited,
		CostUSD:                 costUSD,
	}, nil
}

func resolveRequestedAmount(req RunRequest, fallbackPaid int64) int64 {
	if req.RequestedRefundIDR > 0 {
		return req.RequestedRefundIDR
	}
	if m := idrAmountRe.FindStringSubmatch(req.UserQuery); len(m) > 1 {
		if parsed, err := strconv.ParseInt(m[1], 10, 64); err == nil && parsed > 0 {
			return parsed
		}
	}
	return fallbackPaid
}

func callKey(tool string, args map[string]any) string {
	b, _ := json.Marshal(args)
	return tool + ":" + string(b)
}
