//go:build ignore

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"text/template"
)

type ModelSpec struct {
	Name           string  `json:"name"`
	Provider       string  `json:"provider"`
	InputUSDPer1M  float64 `json:"input_usd_per_1m"`
	OutputUSDPer1M float64 `json:"output_usd_per_1m"`
	PrefillTPS     int     `json:"prefill_tps"`
	DecodeTPS      int     `json:"decode_tps"`
}

type GatewaySpec struct {
	PrimaryModel                string  `json:"primary_model"`
	FallbackModel               string  `json:"fallback_model"`
	SemanticSimilarityThreshold float64 `json:"semantic_similarity_threshold"`
	MaxOutputTokensCap          int     `json:"max_output_tokens_cap"`
	DefaultRPM                  int     `json:"default_rpm"`
	DefaultDailyBudgetUSD       float64 `json:"default_daily_budget_usd"`
	RetryableStatusCodes        []int   `json:"retryable_status_codes"`
	MaxSchemaRepairRetries      int     `json:"max_schema_repair_retries"`
}

type Contract struct {
	SchemaVersion string      `json:"schema_version"`
	Gateway       GatewaySpec `json:"gateway"`
	Models        []ModelSpec `json:"models"`
	Guardrails    struct {
		BlockSemanticCacheOnPIIOrID bool     `json:"block_semantic_cache_on_pii_or_id"`
		EntityPatterns              []string `json:"entity_patterns"`
	} `json:"guardrails"`
}

func canonicalJSON(v any) []byte {
	switch val := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		var buf bytes.Buffer
		buf.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				buf.WriteByte(',')
			}
			kb, _ := json.Marshal(k)
			buf.Write(kb)
			buf.WriteByte(':')
			buf.Write(canonicalJSON(val[k]))
		}
		buf.WriteByte('}')
		return buf.Bytes()
	case []any:
		var buf bytes.Buffer
		buf.WriteByte('[')
		for i, elem := range val {
			if i > 0 {
				buf.WriteByte(',')
			}
			buf.Write(canonicalJSON(elem))
		}
		buf.WriteByte(']')
		return buf.Bytes()
	case json.Number:
		return []byte(val.String())
	default:
		b, _ := json.Marshal(val)
		return b
	}
}

func main() {
	raw, err := os.ReadFile("../../../feature_contract.json")
	if err != nil {
		panic(err)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var generic any
	if err := dec.Decode(&generic); err != nil {
		panic(err)
	}
	sum := sha256.Sum256(canonicalJSON(generic))
	hash := hex.EncodeToString(sum[:])[:12]

	var c Contract
	if err := json.Unmarshal(raw, &c); err != nil {
		panic(err)
	}

	tmpl := template.Must(template.New("contract").Parse(`// Code generated from feature_contract.json; DO NOT EDIT.
package contract

const (
	SchemaVersion               = "{{.SchemaVersion}}"
	SchemaHash                  = "{{.SchemaHash}}"
	PrimaryModel                = "{{.Gateway.PrimaryModel}}"
	FallbackModel               = "{{.Gateway.FallbackModel}}"
	SemanticSimilarityThreshold = {{.Gateway.SemanticSimilarityThreshold}}
	MaxOutputTokensCap          = {{.Gateway.MaxOutputTokensCap}}
	DefaultRPM                  = {{.Gateway.DefaultRPM}}
	DefaultDailyBudgetUSD       = {{.Gateway.DefaultDailyBudgetUSD}}
	MaxSchemaRepairRetries      = {{.Gateway.MaxSchemaRepairRetries}}
	BlockSemanticCacheOnEntity  = {{.Guardrails.BlockSemanticCacheOnPIIOrID}}
)

type ModelPricing struct {
	Name           string
	Provider       string
	InputUSDPer1M  float64
	OutputUSDPer1M float64
}

var Models = map[string]ModelPricing{
{{- range .Models}}
	"{{.Name}}": {
		Name:           "{{.Name}}",
		Provider:       "{{.Provider}}",
		InputUSDPer1M:  {{.InputUSDPer1M}},
		OutputUSDPer1M: {{.OutputUSDPer1M}},
	},
{{- end}}
}

var RetryableStatusCodes = map[int]bool{
{{- range .Gateway.RetryableStatusCodes}}
	{{.}}: true,
{{- end}}
}

func ComputeCostUSD(model string, promptTokens, completionTokens int) float64 {
	m, ok := Models[model]
	if !ok {
		m = Models[PrimaryModel]
	}
	inCost := (float64(promptTokens) / 1_000_000.0) * m.InputUSDPer1M
	outCost := (float64(completionTokens) / 1_000_000.0) * m.OutputUSDPer1M
	return inCost + outCost
}
`))

	out, err := os.Create("contract_gen.go")
	if err != nil {
		panic(err)
	}
	defer out.Close()

	err = tmpl.Execute(out, struct {
		Contract
		SchemaHash string
	}{Contract: c, SchemaHash: hash})
	if err != nil {
		panic(err)
	}
	fmt.Printf("Generated contract_gen.go (schema_hash=%s)\n", hash)
}
