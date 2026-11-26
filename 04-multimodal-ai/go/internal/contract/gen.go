//go:build ignore

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"text/template"
)

// Struct field order matches alphabetical JSON keys from Python json.dumps(..., sort_keys=True, separators=(",", ":"))
type AudioContract struct {
	Channels   int      `json:"channels"`
	FrameMs    int      `json:"frame_ms"`
	HopMs      int      `json:"hop_ms"`
	Labels     []string `json:"labels"`
	NumFrames  int      `json:"num_frames"`
	SampleRate int      `json:"sample_rate"`
}

type VideoContract struct {
	DedupThreshold float64 `json:"dedup_threshold"`
	MaxFrames      int     `json:"max_frames"`
	SampleFPS      int     `json:"sample_fps"`
}

type VisionContract struct {
	Channels       int       `json:"channels"`
	ColorSpace     string    `json:"color_space"`
	EKYCLabels     []string  `json:"ekyc_labels"`
	Height         int       `json:"height"`
	Layout         string    `json:"layout"`
	Mean           []float64 `json:"mean"`
	Std            []float64 `json:"std"`
	UnboxingLabels []string  `json:"unboxing_labels"`
	Width          int       `json:"width"`
}

type Contract struct {
	Audio         AudioContract  `json:"audio"`
	SchemaVersion string         `json:"schema_version"`
	Video         VideoContract  `json:"video"`
	Vision        VisionContract `json:"vision"`
}

func main() {
	raw, err := os.ReadFile("../../../feature_contract.json")
	if err != nil {
		panic(err)
	}
	var c Contract
	if err := json.Unmarshal(raw, &c); err != nil {
		panic(err)
	}

	b, err := json.Marshal(c)
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(b)
	hash := hex.EncodeToString(sum[:])[:12]

	tmpl := template.Must(template.New("gen").Parse(`// Code generated from feature_contract.json; DO NOT EDIT.
package contract

const (
	SchemaVersion = "{{.SchemaVersion}}"
	SchemaHash    = "{{.SchemaHash}}"

	VisionWidth    = {{.Vision.Width}}
	VisionHeight   = {{.Vision.Height}}
	VisionChannels = {{.Vision.Channels}}

	AudioSampleRate = {{.Audio.SampleRate}}
	AudioChannels   = {{.Audio.Channels}}
	AudioFrameMs    = {{.Audio.FrameMs}}
	AudioHopMs      = {{.Audio.HopMs}}
	AudioNumFrames  = {{.Audio.NumFrames}}

	VideoSampleFPS      = {{.Video.SampleFPS}}
	VideoMaxFrames      = {{.Video.MaxFrames}}
	VideoDedupThreshold = {{.Video.DedupThreshold}}
)

var (
	VisionMean = [3]float32{ {{index .Vision.Mean 0}}, {{index .Vision.Mean 1}}, {{index .Vision.Mean 2}} }
	VisionStd  = [3]float32{ {{index .Vision.Std 0}}, {{index .Vision.Std 1}}, {{index .Vision.Std 2}} }

	EKYCLabels = []string{
{{- range .Vision.EKYCLabels}}
		"{{.}}",
{{- end}}
	}
	UnboxingLabels = []string{
{{- range .Vision.UnboxingLabels}}
		"{{.}}",
{{- end}}
	}
	AudioLabels = []string{
{{- range .Audio.Labels}}
		"{{.}}",
{{- end}}
	}
)
`))

	f, err := os.Create("contract_gen.go")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	if err := tmpl.Execute(f, struct {
		Contract
		SchemaHash string
	}{Contract: c, SchemaHash: hash}); err != nil {
		panic(err)
	}
	fmt.Printf("Generated contract_gen.go (schema_hash=%s)\n", hash)
}
