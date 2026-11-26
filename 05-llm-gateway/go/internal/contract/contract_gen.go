// Code generated from feature_contract.json; DO NOT EDIT.
package contract

const (
	SchemaVersion               = "v1.0.0"
	SchemaHash                  = "6b0739238375"
	PrimaryModel                = "gpt-4o-mini"
	FallbackModel               = "gemini-2.5-flash"
	SemanticSimilarityThreshold = 0.75
	MaxOutputTokensCap          = 512
	DefaultRPM                  = 120
	DefaultDailyBudgetUSD       = 5.25
	MaxSchemaRepairRetries      = 1
	BlockSemanticCacheOnEntity  = true
)

type ModelPricing struct {
	Name           string
	Provider       string
	InputUSDPer1M  float64
	OutputUSDPer1M float64
}

var Models = map[string]ModelPricing{
	"gpt-4o-mini": {
		Name:           "gpt-4o-mini",
		Provider:       "openai",
		InputUSDPer1M:  0.15,
		OutputUSDPer1M: 0.6,
	},
	"gemini-2.5-flash": {
		Name:           "gemini-2.5-flash",
		Provider:       "gemini",
		InputUSDPer1M:  0.1,
		OutputUSDPer1M: 0.4,
	},
}

var RetryableStatusCodes = map[int]bool{
	429: true,
	500: true,
	502: true,
	503: true,
	504: true,
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
