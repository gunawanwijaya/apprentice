package viterbi

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"strings"
	"unicode"
)

type Model struct {
	Labels            []string                      `json:"labels"`
	StateWeights      map[string]map[string]float64 `json:"state_weights"`
	TransitionWeights map[string]map[string]float64 `json:"transition_weights"`
}

func Load(path string) (*Model, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m Model
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

func isUpper(s string) bool {
	hasCased := false
	for _, r := range s {
		if unicode.IsLower(r) {
			return false
		}
		if unicode.IsUpper(r) {
			hasCased = true
		}
	}
	return hasCased
}

func isTitle(s string) bool {
	cased := false
	prevCased := false
	for _, r := range s {
		if unicode.IsUpper(r) || unicode.IsTitle(r) {
			if prevCased {
				return false
			}
			cased = true
			prevCased = true
		} else if unicode.IsLower(r) {
			if !prevCased {
				return false
			}
			cased = true
			prevCased = true
		} else {
			prevCased = false
		}
	}
	return cased
}

func isDigit(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

func suffix(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[len(r)-n:])
}

func boolInt(b bool) string {
	if b {
		return "1"
	}
	return "0"
}

func wordFeatures(tokens []string, i int) []string {
	w := tokens[i]
	feats := []string{
		"bias",
		"w.lower=" + strings.ToLower(w),
		"w.suf3=" + suffix(w, 3),
		"w.suf2=" + suffix(w, 2),
		"w.isupper=" + boolInt(isUpper(w)),
		"w.istitle=" + boolInt(isTitle(w)),
		"w.isdigit=" + boolInt(isDigit(w)),
	}
	if i > 0 {
		pw := tokens[i-1]
		feats = append(feats, "-1:w.lower="+strings.ToLower(pw), "-1:w.istitle="+boolInt(isTitle(pw)))
	} else {
		feats = append(feats, "BOS")
	}
	if i < len(tokens)-1 {
		nw := tokens[i+1]
		feats = append(feats, "+1:w.lower="+strings.ToLower(nw), "+1:w.istitle="+boolInt(isTitle(nw)))
	} else {
		feats = append(feats, "EOS")
	}
	return feats
}

func (m *Model) Predict(tokens []string) []string {
	n := len(tokens)
	k := len(m.Labels)
	if n == 0 || k == 0 {
		return nil
	}

	dp := make([][]float64, n)
	back := make([][]int, n)
	for t := 0; t < n; t++ {
		dp[t] = make([]float64, k)
		back[t] = make([]int, k)
		feats := wordFeatures(tokens, t)

		// Emission score for each label at step t
		emit := make([]float64, k)
		for _, f := range feats {
			if lw, ok := m.StateWeights[f]; ok {
				for j, lbl := range m.Labels {
					emit[j] += lw[lbl]
				}
			}
		}

		if t == 0 {
			copy(dp[0], emit)
			continue
		}

		for j, curLbl := range m.Labels {
			bestScore := math.Inf(-1)
			bestPrev := 0
			for p, prevLbl := range m.Labels {
				score := dp[t-1][p] + m.TransitionWeights[prevLbl][curLbl] + emit[j]
				if score > bestScore {
					bestScore = score
					bestPrev = p
				}
			}
			dp[t][j] = bestScore
			back[t][j] = bestPrev
		}
	}

	// Backtrack highest-scoring BIO tag path
	bestLast := 0
	for j := 1; j < k; j++ {
		if dp[n-1][j] > dp[n-1][bestLast] {
			bestLast = j
		}
	}
	tags := make([]string, n)
	cur := bestLast
	for t := n - 1; t >= 0; t-- {
		tags[t] = m.Labels[cur]
		cur = back[t][cur]
	}
	_ = strconv.Itoa // keep import clean if extended
	return tags
}
