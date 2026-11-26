// Code generated from feature_contract.json; DO NOT EDIT.
package contract

import "math"

const (
	SchemaVersion = "v1.1.0"
	SchemaHash    = "fe66b8873d6c"
	Threshold     = float32(0.12)
	FeatureCount  = 5
)

type PredictRequest struct {
	DistanceKm float32 `json:"distance_km"`
	WeightGrams *float32 `json:"weight_grams"`
	SellerLateRate *float32 `json:"seller_late_rate"`
	IsWeekend float32 `json:"is_weekend"`
	FreightRatio float32 `json:"freight_ratio"`
}

func floatOrNaN(v *float32) float32 {
	if v == nil {
		return float32(math.NaN())
	}
	return *v
}

func (r PredictRequest) ToFloat32Slice() []float32 {
	return []float32{
		r.DistanceKm,
		floatOrNaN(r.WeightGrams),
		floatOrNaN(r.SellerLateRate),
		r.IsWeekend,
		r.FreightRatio,
	}
}
