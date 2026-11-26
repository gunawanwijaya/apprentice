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

type Contract struct {
	SchemaVersion string   `json:"schema_version"`
	ModelID       string   `json:"model_id"`
	MaxSeqLen     int      `json:"max_seq_len"`
	DoLowerCase   bool     `json:"do_lower_case"`
	VocabFile     string   `json:"vocab_file"`
	Labels        []string `json:"labels"`
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

	// Keys sorted alphabetically to match Python json.dumps(..., sort_keys=True, separators=(",", ":"))
	type canonical struct {
		DoLowerCase bool     `json:"do_lower_case"`
		Labels      []string `json:"labels"`
		MaxSeqLen   int      `json:"max_seq_len"`
		ModelID     string   `json:"model_id"`
	}
	b, _ := json.Marshal(canonical{
		DoLowerCase: c.DoLowerCase,
		Labels:      c.Labels,
		MaxSeqLen:   c.MaxSeqLen,
		ModelID:     c.ModelID,
	})
	sum := sha256.Sum256(b)
	hash := hex.EncodeToString(sum[:])[:12]

	tmpl := template.Must(template.New("gen").Parse(`// Code generated from feature_contract.json; DO NOT EDIT.
package contract

const (
	SchemaVersion = "{{.SchemaVersion}}"
	SchemaHash    = "{{.SchemaHash}}"
	ModelID       = "{{.ModelID}}"
	MaxSeqLen     = {{.MaxSeqLen}}
	DoLowerCase   = {{.DoLowerCase}}
	VocabFile     = "{{.VocabFile}}"
)

var Labels = []string{
{{- range .Labels}}
	"{{.}}",
{{- end}}
}
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
