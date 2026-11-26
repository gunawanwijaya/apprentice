package ledger

import (
	"errors"
	"sync"

	"golang.org/x/time/rate"
)

var (
	ErrRateLimited    = errors.New("rate limit exceeded (RPM)")
	ErrBudgetExceeded = errors.New("tenant daily USD budget exceeded")
)

type TenantStats struct {
	TenantID         string  `json:"tenant_id"`
	RequestsTotal    int     `json:"requests_total"`
	CacheHits        int     `json:"cache_hits"`
	StreamAborts     int     `json:"stream_aborts"`
	Failovers        int     `json:"failovers"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	SpendUSD         float64 `json:"spend_usd"`
	BudgetUSD        float64 `json:"budget_usd"`
}

type Manager struct {
	mu           sync.Mutex
	defaultRPM   int
	defaultCap   float64
	limiters     map[string]*rate.Limiter
	tenantLedger map[string]*TenantStats
}

func NewManager(defaultRPM int, defaultBudgetUSD float64) *Manager {
	return &Manager{
		defaultRPM:   defaultRPM,
		defaultCap:   defaultBudgetUSD,
		limiters:     make(map[string]*rate.Limiter),
		tenantLedger: make(map[string]*TenantStats),
	}
}

func (m *Manager) SetTenantPolicy(tenantID string, rpm int, budgetUSD float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	burst := rpm / 6
	if burst < 1 {
		burst = 1
	}
	m.limiters[tenantID] = rate.NewLimiter(rate.Limit(float64(rpm)/60.0), burst)
	st := m.ensureLocked(tenantID)
	st.BudgetUSD = budgetUSD
}

func (m *Manager) ensureLocked(tenantID string) *TenantStats {
	st, ok := m.tenantLedger[tenantID]
	if !ok {
		st = &TenantStats{TenantID: tenantID, BudgetUSD: m.defaultCap}
		m.tenantLedger[tenantID] = st
	}
	if _, ok := m.limiters[tenantID]; !ok {
		burst := m.defaultRPM / 6
		if burst < 2 {
			burst = 2
		}
		m.limiters[tenantID] = rate.NewLimiter(rate.Limit(float64(m.defaultRPM)/60.0), burst)
	}
	return st
}

func (m *Manager) CheckAdmission(tenantID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	st := m.ensureLocked(tenantID)
	if st.SpendUSD >= st.BudgetUSD {
		return ErrBudgetExceeded
	}
	if !m.limiters[tenantID].Allow() {
		return ErrRateLimited
	}
	return nil
}

func (m *Manager) Record(tenantID string, promptTok, compTok int, costUSD float64, cacheHit, aborted, failedOver bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	st := m.ensureLocked(tenantID)
	st.RequestsTotal++
	st.PromptTokens += promptTok
	st.CompletionTokens += compTok
	st.SpendUSD += costUSD
	if cacheHit {
		st.CacheHits++
	}
	if aborted {
		st.StreamAborts++
	}
	if failedOver {
		st.Failovers++
	}
}

func (m *Manager) Snapshot(tenantID string) TenantStats {
	m.mu.Lock()
	defer m.mu.Unlock()
	return *m.ensureLocked(tenantID)
}
