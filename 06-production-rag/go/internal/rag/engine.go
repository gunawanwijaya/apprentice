package rag

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"golang.org/x/sync/errgroup"
)

var (
	reNonAlnum = regexp.MustCompile(`[^a-z0-9\-_/\s]+`)
	reCitation = regexp.MustCompile(`\[(POL-[A-Z0-9]+-\d+#\d+)\]`)
	stopWords  = map[string]struct{}{
		"a": {}, "an": {}, "the": {}, "is": {}, "are": {}, "was": {}, "were": {},
		"to": {}, "of": {}, "in": {}, "on": {}, "for": {}, "and": {}, "or": {},
		"i": {}, "my": {}, "me": {}, "you": {}, "your": {}, "we": {}, "our": {},
		"can": {}, "could": {}, "would": {}, "should": {}, "do": {}, "does": {},
		"did": {}, "please": {}, "help": {}, "want": {}, "need": {}, "how": {},
		"what": {}, "when": {}, "where": {}, "with": {}, "that": {}, "this": {},
		"it": {}, "be": {}, "have": {},
	}
	tracer      = otel.Tracer("06-production-rag/engine")
	meter       = otel.Meter("06-production-rag/engine")
	stageDur, _ = meter.Float64Histogram(
		"rag.stage.duration",
		metric.WithUnit("s"),
		metric.WithDescription("Duration of hybrid RAG retrieval stages in seconds"),
		metric.WithExplicitBucketBoundaries(0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1),
	)
)

type Contract struct {
	SchemaVersion string `json:"schema_version"`
	Retrieval     struct {
		EmbedDim           int     `json:"embed_dim"`
		RRFK               int     `json:"rrf_k"`
		BM25TopK           int     `json:"bm25_top_k"`
		VectorTopK         int     `json:"vector_top_k"`
		RerankTopK         int     `json:"rerank_top_k"`
		FinalContextChunks int     `json:"final_context_chunks"`
		MinRelevanceScore  float64 `json:"min_relevance_score"`
		BM25K1             float64 `json:"bm25_k1"`
		BM25B              float64 `json:"bm25_b"`
	} `json:"retrieval"`
	Grounding struct {
		RequireCitations    bool `json:"require_citations"`
		AbstainOnLowScore   bool `json:"abstain_on_low_score"`
		EnforceTenantFilter bool `json:"enforce_tenant_filter"`
	} `json:"grounding"`
}

type Chunk struct {
	ChunkID     string `json:"chunk_id"`
	TenantID    string `json:"tenant_id"`
	PolicyCode  string `json:"policy_code"`
	Intent      string `json:"intent"`
	Header      string `json:"header"`
	Content     string `json:"content"`
	IndexedText string `json:"indexed_text"`
}

type ScoredChunk struct {
	Chunk Chunk   `json:"chunk"`
	Score float64 `json:"score"`
}

type Engine struct {
	mu         sync.RWMutex
	Contract   Contract
	SchemaHash string
	docs       []Chunk
	docByID    map[string]Chunk
	docLens    []int
	avgdl      float64
	tfs        []map[string]int
	idf        map[string]float64
	vecs       [][]float64
}

func LoadContract(path string) (Contract, string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Contract{}, "", err
	}
	var generic any
	if err := json.Unmarshal(raw, &generic); err != nil {
		return Contract{}, "", err
	}
	canon, err := json.Marshal(generic)
	if err != nil {
		return Contract{}, "", err
	}
	sum := sha256.Sum256(canon)
	hash := hex.EncodeToString(sum[:])[:12]

	var c Contract
	if err := json.Unmarshal(raw, &c); err != nil {
		return Contract{}, "", err
	}
	return c, hash, nil
}

func LoadCorpusJSONL(path string) ([]Chunk, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var docs []Chunk
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var c Chunk
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			return nil, err
		}
		docs = append(docs, c)
	}
	return docs, sc.Err()
}

func NewEngine(c Contract, schemaHash string, docs []Chunk) *Engine {
	e := &Engine{
		Contract:   c,
		SchemaHash: schemaHash,
	}
	e.rebuildLocked(docs)
	return e
}

func (e *Engine) UpsertChunk(doc Chunk) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if doc.IndexedText == "" {
		doc.IndexedText = doc.Header + "\n" + doc.Content
	}
	updated := false
	next := make([]Chunk, len(e.docs))
	copy(next, e.docs)
	for i := range next {
		if next[i].ChunkID == doc.ChunkID {
			next[i] = doc
			updated = true
			break
		}
	}
	if !updated {
		next = append(next, doc)
	}
	e.rebuildLocked(next)
}

func (e *Engine) CorpusSize() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.docs)
}

func (e *Engine) rebuildLocked(docs []Chunk) {
	n := len(docs)
	e.docs = docs
	e.docByID = make(map[string]Chunk, n)
	e.docLens = make([]int, n)
	e.tfs = make([]map[string]int, n)
	e.vecs = make([][]float64, n)

	df := make(map[string]int)
	totalLen := 0
	for i, d := range docs {
		e.docByID[d.ChunkID] = d
		toks := Tokenize(d.IndexedText)
		e.docLens[i] = len(toks)
		totalLen += len(toks)
		tf := make(map[string]int, len(toks))
		for _, t := range toks {
			tf[t]++
		}
		e.tfs[i] = tf
		for term := range tf {
			df[term]++
		}
		e.vecs[i] = EmbedText(d.IndexedText, e.Contract.Retrieval.EmbedDim)
	}
	denom := n
	if denom < 1 {
		denom = 1
	}
	e.avgdl = float64(totalLen) / float64(denom)
	e.idf = make(map[string]float64, len(df))
	for term, freq := range df {
		e.idf[term] = math.Log(1.0 + (float64(n-freq)+0.5)/(float64(freq)+0.5))
	}
}

func Tokenize(text string) []string {
	clean := reNonAlnum.ReplaceAllString(strings.ToLower(text), " ")
	raw := strings.Fields(clean)
	tokens := make([]string, 0, len(raw))
	for _, t := range raw {
		if len(t) <= 1 {
			continue
		}
		if _, isStop := stopWords[t]; isStop {
			continue
		}
		tokens = append(tokens, t)
	}
	if len(tokens) == 0 {
		return raw
	}
	return tokens
}

func fnv1a32(s string) uint32 {
	var h uint32 = 2166136261
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= 16777619
	}
	return h
}

func EmbedText(text string, dim int) []float64 {
	vec := make([]float64, dim)
	tokens := Tokenize(text)
	uDim := uint32(dim)
	for i, tok := range tokens {
		stem := tok
		for _, suf := range []string{"ing", "ed", "s"} {
			if len(stem) > len(suf)+2 && strings.HasSuffix(stem, suf) {
				stem = stem[:len(stem)-len(suf)]
				break
			}
		}
		vec[fnv1a32("w:"+stem)%uDim] += 1.5
		if i+1 < len(tokens) {
			vec[fnv1a32("b:"+stem+"_"+tokens[i+1])%uDim] += 1.0
		}
		for j := 0; j < len(stem)-2; j++ {
			vec[fnv1a32("c:"+stem[j:j+3])%uDim] += 0.5
		}
	}
	var sumSq float64
	for _, v := range vec {
		sumSq += v * v
	}
	if norm := math.Sqrt(sumSq); norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}
	return vec
}

type hit struct {
	id    string
	score float64
}

func sortHits(hits []hit, topK int) []hit {
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].score != hits[j].score {
			return hits[i].score > hits[j].score
		}
		return hits[i].id < hits[j].id
	})
	if len(hits) > topK {
		return hits[:topK]
	}
	return hits
}

func (e *Engine) searchBM25(ctx context.Context, query, tenantID string, topK int) []hit {
	start := time.Now()
	ctx, span := tracer.Start(ctx, "rag.bm25")
	defer func() {
		stageDur.Record(ctx, time.Since(start).Seconds(), metric.WithAttributes(attribute.String("stage", "bm25")))
		span.End()
	}()

	e.mu.RLock()
	defer e.mu.RUnlock()

	qTokens := Tokenize(query)
	k1 := e.Contract.Retrieval.BM25K1
	b := e.Contract.Retrieval.BM25B
	var scored []hit

	for i, doc := range e.docs {
		if doc.TenantID != tenantID {
			continue
		}
		tf := e.tfs[i]
		dl := float64(e.docLens[i])
		var score float64
		for _, qt := range qTokens {
			f, ok := tf[qt]
			if !ok {
				continue
			}
			idf := e.idf[qt]
			ff := float64(f)
			score += idf * (ff * (k1 + 1.0)) / (ff + k1*(1.0-b+b*dl/e.avgdl))
		}
		if score > 0 {
			scored = append(scored, hit{id: doc.ChunkID, score: score})
		}
	}
	return sortHits(scored, topK)
}

func (e *Engine) searchVector(ctx context.Context, query, tenantID string, topK int) []hit {
	start := time.Now()
	ctx, span := tracer.Start(ctx, "rag.vector")
	defer func() {
		stageDur.Record(ctx, time.Since(start).Seconds(), metric.WithAttributes(attribute.String("stage", "vector")))
		span.End()
	}()

	e.mu.RLock()
	defer e.mu.RUnlock()

	qVec := EmbedText(query, e.Contract.Retrieval.EmbedDim)
	var scored []hit
	for i, doc := range e.docs {
		if doc.TenantID != tenantID {
			continue
		}
		var dot float64
		dv := e.vecs[i]
		for j := range qVec {
			dot += dv[j] * qVec[j]
		}
		scored = append(scored, hit{id: doc.ChunkID, score: dot})
	}
	return sortHits(scored, topK)
}

// Retrieve runs BM25 and Vector search in parallel via errgroup and fuses them with RRF.
func (e *Engine) Retrieve(ctx context.Context, query, tenantID string) ([]ScoredChunk, error) {
	start := time.Now()
	ctx, span := tracer.Start(ctx, "rag.retrieve")
	defer func() {
		stageDur.Record(ctx, time.Since(start).Seconds(), metric.WithAttributes(attribute.String("stage", "rrf_retrieve")))
		span.End()
	}()
	span.SetAttributes(attribute.String("rag.tenant_id", tenantID))

	if e.Contract.Grounding.EnforceTenantFilter && strings.TrimSpace(tenantID) == "" {
		return nil, fmt.Errorf("tenant_id is required")
	}

	var bm25Hits, vecHits []hit
	g, gCtx := errgroup.WithContext(ctx)
	g.Go(func() error {
		bm25Hits = e.searchBM25(gCtx, query, tenantID, e.Contract.Retrieval.BM25TopK)
		return nil
	})
	g.Go(func() error {
		vecHits = e.searchVector(gCtx, query, tenantID, e.Contract.Retrieval.VectorTopK)
		return nil
	})
	if err := g.Wait(); err != nil {
		return nil, err
	}

	k := float64(e.Contract.Retrieval.RRFK)
	scores := make(map[string]float64)
	for rank, h := range bm25Hits {
		scores[h.id] += 1.0 / (k + float64(rank+1))
	}
	for rank, h := range vecHits {
		scores[h.id] += 1.0 / (k + float64(rank+1))
	}

	fused := make([]hit, 0, len(scores))
	for id, s := range scores {
		fused = append(fused, hit{id: id, score: s})
	}
	fused = sortHits(fused, e.Contract.Retrieval.RerankTopK)

	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]ScoredChunk, 0, len(fused))
	for _, h := range fused {
		out = append(out, ScoredChunk{Chunk: e.docByID[h.id], Score: h.score})
	}
	span.SetAttributes(attribute.Int("rag.hits", len(out)))
	return out, nil
}

// ShouldAbstain checks whether retrieval confidence falls below contract min_relevance_score.
func (e *Engine) ShouldAbstain(hits []ScoredChunk) bool {
	if !e.Contract.Grounding.AbstainOnLowScore {
		return false
	}
	if len(hits) == 0 {
		return true
	}
	return hits[0].Score < e.Contract.Retrieval.MinRelevanceScore
}

// VerifyCitations checks that every [POL-XXXX-NN#0] citation in answer exists in the retrieved chunks.
func VerifyCitations(answer string, allowed []ScoredChunk) (valid bool, cited []string, invalid []string) {
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, sc := range allowed {
		allowedSet[sc.Chunk.ChunkID] = struct{}{}
	}
	seen := make(map[string]struct{})
	matches := reCitation.FindAllStringSubmatch(answer, -1)
	for _, m := range matches {
		cid := m[1]
		if _, dup := seen[cid]; dup {
			continue
		}
		seen[cid] = struct{}{}
		cited = append(cited, cid)
		if _, ok := allowedSet[cid]; !ok {
			invalid = append(invalid, cid)
		}
	}
	return len(cited) > 0 && len(invalid) == 0, cited, invalid
}
