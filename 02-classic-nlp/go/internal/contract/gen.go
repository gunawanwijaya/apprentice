//go:build ignore

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"text/template"
)

type Contract struct {
	SchemaVersion string `json:"schema_version"`
	TokenPattern  string `json:"token_pattern"`
	BM25          struct {
		K1   float64 `json:"k1"`
		B    float64 `json:"b"`
		TopK int     `json:"top_k"`
	} `json:"bm25"`
}

func main() {
	raw, err := os.ReadFile("../../../feature_contract.json")
	if err != nil {
		panic(err)
	}

	// Canonical JSON matching Python's json.dumps(contract, sort_keys=True, separators=(",", ":"))
	var generic any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&generic); err != nil {
		panic(err)
	}
	canonBytes, err := json.Marshal(generic)
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(canonBytes)
	hash := hex.EncodeToString(sum[:])[:12]

	var c Contract
	if err := json.Unmarshal(raw, &c); err != nil {
		panic(err)
	}

	tmpl := template.Must(template.New("gen").Parse(`// Code generated from feature_contract.json; DO NOT EDIT.
package contract

const (
	SchemaVersion = "{{.SchemaVersion}}"
	SchemaHash    = "{{.SchemaHash}}"
	TokenPattern  = ` + "`{{.TokenPattern}}`" + `
	BM25K1        = {{.BM25.K1}}
	BM25B         = {{.BM25.B}}
	BM25TopK      = {{.BM25.TopK}}
)
`))

	f, err := os.Create("contract_gen.go")
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
	fmt.Printf("Generated contract_gen.go (schema_hash=%s)\n", hash)
}
