package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// GeminiNativeClient speaks Google's native generateContent REST API using x-goog-api-key,
// avoiding the 401 ACCESS_TOKEN_TYPE_UNSUPPORTED error on AQ.* keys over Bearer auth.
type GeminiNativeClient struct {
	APIKey        string
	ModelOverride string
	HTTPClient    *http.Client
}

func (g *GeminiNativeClient) Complete(ctx context.Context, model string, req ChatRequest) (ChatResult, error) {
	wireModel := model
	if g.ModelOverride != "" {
		wireModel = g.ModelOverride
	}

	var sysParts []map[string]string
	var contents []map[string]any
	for _, m := range req.Messages {
		if m.Role == "system" {
			sysParts = append(sysParts, map[string]string{"text": m.Content})
			continue
		}
		role := "user"
		if m.Role == "assistant" {
			role = "model"
		}
		contents = append(contents, map[string]any{
			"role":  role,
			"parts": []map[string]string{{"text": m.Content}},
		})
	}

	genCfg := map[string]any{
		"temperature": req.Temperature,
	}
	if req.MaxTokens > 0 {
		genCfg["maxOutputTokens"] = req.MaxTokens
	}
	if req.ResponseSchema != "" {
		genCfg["responseMimeType"] = "application/json"
	}

	payload := map[string]any{
		"contents":         contents,
		"generationConfig": genCfg,
	}
	if len(sysParts) > 0 {
		payload["systemInstruction"] = map[string]any{"parts": sysParts}
	}

	body, _ := json.Marshal(payload)
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent", wireModel)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return ChatResult{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", g.APIKey) // AQ.* keys require x-goog-api-key without Authorization: Bearer

	resp, err := g.HTTPClient.Do(httpReq)
	if err != nil {
		return ChatResult{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return ChatResult{}, &UpstreamError{StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(raw))}
	}

	var parsed struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
			FinishReason string `json:"finishReason"`
		} `json:"candidates"`
		UsageMetadata struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
		} `json:"usageMetadata"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return ChatResult{}, err
	}
	if len(parsed.Candidates) == 0 || len(parsed.Candidates[0].Content.Parts) == 0 {
		return ChatResult{}, errors.New("empty gemini candidates")
	}

	finish := "stop"
	if parsed.Candidates[0].FinishReason == "MAX_TOKENS" {
		finish = "length"
	}
	return ChatResult{
		Model:            wireModel,
		Provider:         "gemini",
		Content:          parsed.Candidates[0].Content.Parts[0].Text,
		FinishReason:     finish,
		PromptTokens:     parsed.UsageMetadata.PromptTokenCount,
		CompletionTokens: parsed.UsageMetadata.CandidatesTokenCount,
	}, nil
}

func (g *GeminiNativeClient) Stream(ctx context.Context, model string, req ChatRequest, onDelta func(StreamDelta) error) error {
	res, err := g.Complete(ctx, model, req)
	if err != nil {
		return err
	}
	if err := onDelta(StreamDelta{Token: res.Content, PromptTokens: res.PromptTokens, CompletionTokens: res.CompletionTokens}); err != nil {
		return err
	}
	return onDelta(StreamDelta{Done: true})
}
