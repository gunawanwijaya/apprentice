package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"hash/fnv"
	"math"
	"regexp"
	"strings"
	"sync"
)

const embedDim = 256

var (
	reNonAlpha = regexp.MustCompile(`[^a-z0-9\s]+`)
	stopWords  = map[string]bool{
		"a": true, "an": true, "the": true, "is": true, "are": true, "to": true,
		"do": true, "does": true, "i": true, "my": true, "me": true, "please": true,
		"can": true, "you": true, "tell": true, "what": true, "how": true, "for": true,
	}
)

type CachedResponse struct {
	Content          string `json:"content"`
	Model            string `json:"model"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
}

type semanticEntry struct {
	tenantID string
	vec      [embedDim]float64
	resp     CachedResponse
}

type Store struct {
	mu        sync.RWMutex
	exact     map[string]CachedResponse
	semantic  []semanticEntry
	threshold float64
}

func NewStore(similarityThreshold float64) *Store {
	return &Store{
		exact:     make(map[string]CachedResponse),
		threshold: similarityThreshold,
	}
}

func ExactKey(tenantID, model string, temp float64, canonicalPrompt, schemaName string) string {
	h := sha256.Sum256([]byte(tenantID + "|" + model + "|" + formatTemp(temp) + "|" + schemaName + "|" + strings.TrimSpace(canonicalPrompt)))
	return hex.EncodeToString(h[:])
}

func formatTemp(t float64) string {
	if t <= 0.0001 {
		return "0"
	}
	return "non_zero"
}

func EmbedText(text string) [embedDim]float64 {
	var vec [embedDim]float64
	clean := reNonAlpha.ReplaceAllString(strings.ToLower(text), " ")
	rawTokens := strings.Fields(clean)
	tokens := make([]string, 0, len(rawTokens))
	for _, tok := range rawTokens {
		if !stopWords[tok] {
			tokens = append(tokens, tok)
		}
	}
	if len(tokens) == 0 {
		tokens = rawTokens
	}

	addFeat := func(feat string, weight float64) {
		h := fnv.New32a()
		_, _ = h.Write([]byte(feat))
		idx := h.Sum32() % embedDim
		vec[idx] += weight
	}

	for i, tok := range tokens {
		stem := strings.TrimSuffix(strings.TrimSuffix(strings.TrimSuffix(tok, "ing"), "ed"), "s")
		if len(stem) >= 3 {
			tok = stem
		}
		addFeat("w:"+tok, 1.5)
		if i+1 < len(tokens) {
			addFeat("b:"+tok+"_"+tokens[i+1], 1.0)
		}
		for j := 0; j+3 <= len(tok); j++ {
			addFeat("c:"+tok[j:j+3], 0.5)
		}
	}

	var norm float64
	for _, v := range vec {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}
	return vec
}

func cosine(a, b [embedDim]float64) float64 {
	var dot float64
	for i := 0; i < embedDim; i++ {
		dot += a[i] * b[i]
	}
	return dot
}

func (s *Store) Lookup(tenantID, exactKey, userText string, allowExact, allowSemantic bool) (CachedResponse, string, float64, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if allowExact {
		if hit, ok := s.exact[exactKey]; ok {
			return hit, "exact_sha256", 1.0, true
		}
	}

	if allowSemantic {
		qVec := EmbedText(userText)
		bestScore := -1.0
		var best CachedResponse
		for _, entry := range s.semantic {
			if entry.tenantID != tenantID {
				continue
			}
			score := cosine(qVec, entry.vec)
			if score > bestScore {
				bestScore = score
				best = entry.resp
			}
		}
		if bestScore >= s.threshold {
			return best, "semantic_vector", bestScore, true
		}
	}

	return CachedResponse{}, "miss", 0.0, false
}

func (s *Store) Put(tenantID, exactKey, userText string, resp CachedResponse, storeExact, storeSemantic bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if storeExact {
		s.exact[exactKey] = resp
	}
	if storeSemantic {
		s.semantic = append(s.semantic, semanticEntry{
			tenantID: tenantID,
			vec:      EmbedText(userText),
			resp:     resp,
		})
	}
}
