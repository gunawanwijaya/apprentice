//go:build ignore

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"text/template"
)

type Feature struct {
	Name     string `json:"name"`
	Nullable bool   `json:"nullable"`
	Type     string `json:"type"`
}

type Contract struct {
	SchemaVersion string    `json:"schema_version"`
	Threshold     float32   `json:"threshold"`
	Features      []Feature `json:"features"`
}

func toPascal(s string) string {
	parts := strings.Split(s, "_")
	for i, p := range parts {
		if len(p) > 0 {
			parts[i] = strings.ToUpper(p[:1]) + p[1:]
		}
	}
	return strings.Join(parts, "")
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

	// Canonical JSON matches Python's json.dumps(..., sort_keys=True, separators=(",", ":"))
	type canonicalFeature struct {
		Name     string `json:"name"`
		Nullable bool   `json:"nullable"`
		Type     string `json:"type"`
	}
	cf := make([]canonicalFeature, len(c.Features))
	for i, f := range c.Features {
		cf[i] = canonicalFeature{Name: f.Name, Nullable: f.Nullable, Type: f.Type}
	}
	canonBytes, _ := json.Marshal(cf)
	sum := sha256.Sum256(canonBytes)
	hash := hex.EncodeToString(sum[:])[:12]

	tmpl := template.Must(template.New("gen").Funcs(template.FuncMap{
		"pascal": toPascal,
	}).Parse(`// Code generated from feature_contract.json; DO NOT EDIT.
package contract

import "math"

const (
	SchemaVersion = "{{.SchemaVersion}}"
	SchemaHash    = "{{.SchemaHash}}"
	Threshold     = float32({{.Threshold}})
	FeatureCount  = {{len .Features}}
)

type PredictRequest struct {
{{- range .Features}}
	{{pascal .Name}} {{if .Nullable}}*float32{{else}}float32{{end}} ` + "`json:\"{{.Name}}\"`" + `
{{- end}}
}

func floatOrNaN(v *float32) float32 {
	if v == nil {
		return float32(math.NaN())
	}
	return *v
}

func (r PredictRequest) ToFloat32Slice() []float32 {
	return []float32{
{{- range .Features}}
		{{if .Nullable}}floatOrNaN(r.{{pascal .Name}}){{else}}r.{{pascal .Name}}{{end}},
{{- end}}
	}
}
`))

	f, err := os.Create("features_gen.go")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	data := struct {
		Contract
		SchemaHash string
	}{Contract: c, SchemaHash: hash}

	if err := tmpl.Execute(f, data); err != nil {
		panic(err)
	}
	fmt.Printf("Generated features_gen.go (schema_hash=%s)\n", hash)
}
