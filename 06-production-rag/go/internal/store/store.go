package store

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/gunawanwijaya/apprentice/06-production-rag/go/internal/rag"
	"github.com/jackc/pgx/v5/pgxpool"
)

type PGStore struct {
	pool     *pgxpool.Pool
	embedDim int
}

func NewPGStore(ctx context.Context, dsn string, embedDim int) (*PGStore, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, err
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	s := &PGStore{pool: pool, embedDim: embedDim}
	if err := s.migrate(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	return s, nil
}

func (s *PGStore) Close() { s.pool.Close() }

func (s *PGStore) migrate(ctx context.Context) error {
	ddl := fmt.Sprintf(`
		CREATE EXTENSION IF NOT EXISTS vector;
		CREATE TABLE IF NOT EXISTS rag_chunks (
			chunk_id     TEXT PRIMARY KEY,
			tenant_id    TEXT NOT NULL,
			policy_code  TEXT NOT NULL,
			intent       TEXT NOT NULL,
			header       TEXT NOT NULL,
			content      TEXT NOT NULL,
			indexed_text TEXT NOT NULL,
			embedding    vector(%d) NOT NULL,
			tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', indexed_text)) STORED,
			updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
		);
		CREATE INDEX IF NOT EXISTS idx_rag_chunks_tenant ON rag_chunks (tenant_id);
		CREATE INDEX IF NOT EXISTS idx_rag_chunks_tsv ON rag_chunks USING GIN (tsv);
		CREATE INDEX IF NOT EXISTS idx_rag_chunks_hnsw ON rag_chunks USING hnsw (embedding vector_cosine_ops);
	`, s.embedDim)
	_, err := s.pool.Exec(ctx, ddl)
	return err
}

func (s *PGStore) UpsertChunk(ctx context.Context, c rag.Chunk) error {
	if c.IndexedText == "" {
		c.IndexedText = c.Header + "\n" + c.Content
	}
	vec := rag.EmbedText(c.IndexedText, s.embedDim)
	_, err := s.pool.Exec(ctx, `
		INSERT INTO rag_chunks (chunk_id, tenant_id, policy_code, intent, header, content, indexed_text, embedding)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector)
		ON CONFLICT (chunk_id) DO UPDATE SET
			tenant_id    = EXCLUDED.tenant_id,
			policy_code  = EXCLUDED.policy_code,
			intent       = EXCLUDED.intent,
			header       = EXCLUDED.header,
			content      = EXCLUDED.content,
			indexed_text = EXCLUDED.indexed_text,
			embedding    = EXCLUDED.embedding,
			updated_at   = now()
	`, c.ChunkID, c.TenantID, c.PolicyCode, c.Intent, c.Header, c.Content, c.IndexedText, formatVector(vec))
	return err
}

func (s *PGStore) LoadAll(ctx context.Context) ([]rag.Chunk, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT chunk_id, tenant_id, policy_code, intent, header, content, indexed_text
		FROM rag_chunks
		ORDER BY chunk_id
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []rag.Chunk
	for rows.Next() {
		var c rag.Chunk
		if err := rows.Scan(&c.ChunkID, &c.TenantID, &c.PolicyCode, &c.Intent, &c.Header, &c.Content, &c.IndexedText); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

func formatVector(v []float64) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, x := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(x, 'f', 6, 64))
	}
	b.WriteByte(']')
	return b.String()
}

type S3Store struct {
	client *s3.Client
	bucket string
}

func NewS3Store(ctx context.Context, endpoint, accessKey, secretKey, bucket string) (*S3Store, error) {
	cfg, err := awsconfig.LoadDefaultConfig(ctx,
		awsconfig.WithRegion("us-east-1"),
		awsconfig.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(accessKey, secretKey, "")),
	)
	if err != nil {
		return nil, err
	}
	client := s3.NewFromConfig(cfg, func(o *s3.Options) {
		o.BaseEndpoint = aws.String(endpoint)
		o.UsePathStyle = true
	})
	// Ensure bucket exists (ignore AlreadyOwnedByYou / BucketAlreadyExists)
	_, _ = client.CreateBucket(ctx, &s3.CreateBucketInput{Bucket: aws.String(bucket)})
	return &S3Store{client: client, bucket: bucket}, nil
}

func (s *S3Store) ArchiveChunk(ctx context.Context, c rag.Chunk) error {
	raw, err := json.Marshal(c)
	if err != nil {
		return err
	}
	key := fmt.Sprintf("tenants/%s/%s.json", strings.ToUpper(c.TenantID), c.ChunkID)
	_, err = s.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:      aws.String(s.bucket),
		Key:         aws.String(key),
		Body:        bytes.NewReader(raw),
		ContentType: aws.String("application/json"),
	})
	return err
}
