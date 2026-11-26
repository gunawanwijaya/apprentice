package pii

import "regexp"

var (
	reEmail = regexp.MustCompile(`[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}`)
	reKTP   = regexp.MustCompile(`\b\d{16}\b`)
	rePhone = regexp.MustCompile(`(?:\+?62|0)8[1-9][0-9]{7,11}\b`)
)

// Redact masks structured Indonesian PII (emails, 16-digit NIK/KTP, 08/628 phones) deterministically.
func Redact(text string) string {
	out := reEmail.ReplaceAllString(text, "[EMAIL]")
	out = reKTP.ReplaceAllString(out, "[KTP]")
	out = rePhone.ReplaceAllString(out, "[PHONE]")
	return out
}
