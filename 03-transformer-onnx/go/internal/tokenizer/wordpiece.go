package tokenizer

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"unicode"
)

type WordPiece struct {
	vocab       map[string]int64
	unkID       int64
	clsID       int64
	sepID       int64
	padID       int64
	doLowerCase bool
	maxWordLen  int
}

func Load(vocabPath string, doLowerCase bool) (*WordPiece, error) {
	f, err := os.Open(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("open vocab: %w", err)
	}
	defer f.Close()

	vocab := make(map[string]int64, 32000)
	scanner := bufio.NewScanner(f)
	var idx int64
	for scanner.Scan() {
		vocab[scanner.Text()] = idx
		idx++
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return &WordPiece{
		vocab:       vocab,
		unkID:       vocab["[UNK]"],
		clsID:       vocab["[CLS]"],
		sepID:       vocab["[SEP]"],
		padID:       vocab["[PAD]"],
		doLowerCase: doLowerCase,
		maxWordLen:  200,
	}, nil
}

// isBertPunct matches HuggingFace BasicTokenizer._is_punctuation.
func isBertPunct(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) ||
		(r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
}

func (wp *WordPiece) basicTokenize(text string) []string {
	if wp.doLowerCase {
		text = strings.ToLower(text)
	}
	var out []string
	for _, field := range strings.Fields(text) {
		var buf strings.Builder
		for _, r := range field {
			if isBertPunct(r) {
				if buf.Len() > 0 {
					out = append(out, buf.String())
					buf.Reset()
				}
				out = append(out, string(r))
			} else {
				buf.WriteRune(r)
			}
		}
		if buf.Len() > 0 {
			out = append(out, buf.String())
		}
	}
	return out
}

func (wp *WordPiece) splitWordPiece(word string) []int64 {
	runes := []rune(word)
	if len(runes) > wp.maxWordLen {
		return []int64{wp.unkID}
	}
	var ids []int64
	start := 0
	for start < len(runes) {
		end := len(runes)
		var matchedID int64
		found := false
		for start < end {
			sub := string(runes[start:end])
			if start > 0 {
				sub = "##" + sub
			}
			if id, ok := wp.vocab[sub]; ok {
				matchedID = id
				found = true
				break
			}
			end--
		}
		if !found {
			return []int64{wp.unkID}
		}
		ids = append(ids, matchedID)
		start = end
	}
	return ids
}

// Encode returns fixed-length (input_ids, attention_mask) matching HF padding="max_length", truncation=True.
func (wp *WordPiece) Encode(text string, maxSeqLen int) ([]int64, []int64) {
	tokens := make([]int64, 0, maxSeqLen)
	tokens = append(tokens, wp.clsID)

	for _, w := range wp.basicTokenize(text) {
		for _, id := range wp.splitWordPiece(w) {
			if len(tokens) >= maxSeqLen-1 {
				break
			}
			tokens = append(tokens, id)
		}
		if len(tokens) >= maxSeqLen-1 {
			break
		}
	}
	tokens = append(tokens, wp.sepID)

	inputIDs := make([]int64, maxSeqLen)
	attnMask := make([]int64, maxSeqLen)
	for i := 0; i < len(tokens); i++ {
		inputIDs[i] = tokens[i]
		attnMask[i] = 1
	}
	for i := len(tokens); i < maxSeqLen; i++ {
		inputIDs[i] = wp.padID
		attnMask[i] = 0
	}
	return inputIDs, attnMask
}
