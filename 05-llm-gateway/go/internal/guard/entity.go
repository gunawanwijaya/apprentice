package guard

import (
	"regexp"
	"strings"
)

var (
	reOrderOrTicket = regexp.MustCompile(`(?i)(#\d{3,}|\b(?:ord|inv|trx|tkt|pkg|awb)[-_]?\d{3,}\b)`)
	reEmail         = regexp.MustCompile(`(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b`)
	rePhone         = regexp.MustCompile(`(?:\+62|62|08)[0-9\- ]{7,14}`)
	reLongDigits    = regexp.MustCompile(`\b\d{12,19}\b`)
	reCurrency      = regexp.MustCompile(`(?i)(?:\$\s?\d+(?:[.,]\d+)?|\b(?:idr|rp|usd|jpy)\s?\d[\d.,]*|\b\d+(?:[.,]\d+)?\s?(?:rb|ribu|jt|juta|usd)\b)`)
)

type ScanResult struct {
	EntitySensitive bool     `json:"entity_sensitive"`
	MatchedTags     []string `json:"matched_tags"`
}

// Inspect scans prompt text in microseconds to decide whether semantic caching is safe.
func Inspect(text string) ScanResult {
	var tags []string
	if reOrderOrTicket.MatchString(text) {
		tags = append(tags, "order_or_ticket_id")
	}
	if reEmail.MatchString(text) {
		tags = append(tags, "email_address")
	}
	if rePhone.MatchString(text) {
		tags = append(tags, "phone_number")
	}
	if reLongDigits.MatchString(strings.ReplaceAll(text, "-", "")) {
		tags = append(tags, "nik_or_card_digits")
	}
	if reCurrency.MatchString(text) {
		tags = append(tags, "currency_amount")
	}
	return ScanResult{
		EntitySensitive: len(tags) > 0,
		MatchedTags:     tags,
	}
}
