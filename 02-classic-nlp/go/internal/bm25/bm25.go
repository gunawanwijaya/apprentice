package bm25

import (
	"encoding/json"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"
)

type Doc struct {
	ID       string   `json:"id"`
	Text     string   `json:"text"`
	Response string   `json:"response"`
	Tokens   []string `json:"tokens"`
}

type CorpusFile struct {
	K1   float64 `json:"k1"`
	B    float64 `json:"b"`
	Docs []Doc   `json:"docs"`
}

type Result struct {
	ID       string  `json:"id"`
	Score    float64 `json:"score"`
	Response string  `json:"response"`
}

type Index struct {
	k1    float64
	b     float64
	avgDL float64
	docs  []Doc
	tfs   []map[string]int
	idf   map[string]float64
	tokRe *regexp.Regexp
}

func Load(path, tokenPattern string) (*Index, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cf CorpusFile
	if err := json.Unmarshal(raw, &cf); err != nil {
		return nil, err
	}
	re, err := regexp.Compile(tokenPattern)
	if err != nil {
		return nil, err
	}

	n := float64(len(cf.Docs))
	df := make(map[string]int)
	tfs := make([]map[string]int, len(cf.Docs))
	totalLen := 0

	for i, d := range cf.Docs {
		totalLen += len(d.Tokens)
		tf := make(map[string]int, len(d.Tokens))
		for _, tok := range d.Tokens {
			tf[tok]++
		}
		for tok := range tf {
			df[tok]++
		}
		tfs[i] = tf
	}

	idf := make(map[string]float64, len(df))
	idfSum := 0.0
	var negTerms []string
	for tok, freq := range df {
		v := math.Log(n-float64(freq)+0.5) - math.Log(float64(freq)+0.5)
		idf[tok] = v
		idfSum += v
		if v < 0 {
			negTerms = append(negTerms, tok)
		}
	}
	avgIDF := idfSum / math.Max(float64(len(idf)), 1.0)
	for _, tok := range negTerms {
		idf[tok] = 0.25 * avgIDF
	}

	return &Index{
		k1:    cf.K1,
		b:     cf.B,
		avgDL: float64(totalLen) / math.Max(n, 1.0),
		docs:  cf.Docs,
		tfs:   tfs,
		idf:   idf,
		tokRe: re,
	}, nil
}

func (idx *Index) Search(query string, topK int) []Result {
	qTokens := idx.tokRe.FindAllString(strings.ToLower(query), -1)
	scores := make([]Result, len(idx.docs))

	for i, d := range idx.docs {
		dl := float64(len(d.Tokens))
		denomNorm := idx.k1 * (1.0 - idx.b + idx.b*(dl/idx.avgDL))
		score := 0.0
		for _, qt := range qTokens {
			tf := float64(idx.tfs[i][qt])
			if tf == 0 {
				continue
			}
			score += idx.idf[qt] * (tf * (idx.k1 + 1.0)) / (tf + denomNorm)
		}
		scores[i] = Result{ID: d.ID, Score: score, Response: d.Response}
	}

	sort.SliceStable(scores, func(i, j int) bool {
		return scores[i].Score > scores[j].Score
	})

	out := make([]Result, 0, topK)
	for _, r := range scores {
		if r.Score <= 0 || len(out) >= topK {
			break
		}
		out = append(out, r)
	}
	return out
}
