package evalguard

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// Regression — deep audit 2026-07-25 MEDIUM (availability), finding M262.
//
// handleErrorResponse used to build RateLimitError.RetryAfter straight from the
// Retry-After header with NO upper bound:
//
//	retryAfter := 60 * time.Second
//	if ra := resp.Header.Get("Retry-After"); ra != "" {
//	    if seconds, err := strconv.Atoi(ra); err == nil && seconds > 0 {
//	        retryAfter = time.Duration(seconds) * time.Second
//	    }
//	}
//
// and doRaw slept for exactly that. `Retry-After: 3600` therefore blocked a
// single SDK call for an hour per attempt, with WithTimeout(30s) configured —
// c.httpClient.Timeout bounds each exchange, never the sleep. These tests pin
// the ceiling, the HTTP-date form, and the presence of jitter.

func TestParseError_ClampsHostileRetryAfter(t *testing.T) {
	resp := &http.Response{
		StatusCode: 429,
		Header:     http.Header{"Retry-After": []string{"3600"}},
	}
	c := &Client{}
	err := c.handleErrorResponse(resp, []byte(`{"error":{"message":"slow down"}}`), "req-1")
	rl, ok := err.(*RateLimitError)
	if !ok {
		t.Fatalf("expected *RateLimitError, got %T", err)
	}
	if rl.RetryAfter > maxRetryDelay {
		t.Fatalf("RetryAfter %v exceeds ceiling %v — a hostile header can park the caller", rl.RetryAfter, maxRetryDelay)
	}
	if rl.RetryAfter >= time.Hour {
		t.Fatalf("RetryAfter %v honoured verbatim; the pre-fix bug is back", rl.RetryAfter)
	}
}

func TestParseError_HonoursShortRetryAfter(t *testing.T) {
	resp := &http.Response{
		StatusCode: 429,
		Header:     http.Header{"Retry-After": []string{"2"}},
	}
	c := &Client{}
	rl := c.handleErrorResponse(resp, nil, "req-1").(*RateLimitError)
	if rl.RetryAfter != 2*time.Second {
		t.Fatalf("a short hint must be honoured exactly, got %v", rl.RetryAfter)
	}
}

func TestParseError_UnderstandsHTTPDateRetryAfter(t *testing.T) {
	at := time.Now().Add(4 * time.Second).UTC().Format(http.TimeFormat)
	resp := &http.Response{
		StatusCode: 429,
		Header:     http.Header{"Retry-After": []string{at}},
	}
	c := &Client{}
	rl := c.handleErrorResponse(resp, nil, "req-1").(*RateLimitError)
	// strconv.Atoi errors on the date form, so this used to silently fall back
	// to the 60s default.
	if rl.RetryAfter <= 0 || rl.RetryAfter > 5*time.Second {
		t.Fatalf("HTTP-date Retry-After not understood: got %v", rl.RetryAfter)
	}
}

func TestParseError_NoHeaderUsesBoundedFallback(t *testing.T) {
	resp := &http.Response{StatusCode: 429, Header: http.Header{}}
	c := &Client{}
	rl := c.handleErrorResponse(resp, nil, "req-1").(*RateLimitError)
	if rl.RetryAfter > maxRetryDelay {
		t.Fatalf("absent-header fallback %v exceeds ceiling %v", rl.RetryAfter, maxRetryDelay)
	}
}

func TestClampAndJitter_AlwaysBounded(t *testing.T) {
	for _, d := range []time.Duration{-1, 0, time.Millisecond, time.Second, time.Hour} {
		if got := clampRetryDelay(d); got > maxRetryDelay || got <= 0 {
			t.Fatalf("clampRetryDelay(%v) = %v, out of bounds", d, got)
		}
		if got := jitteredDelay(d); got > maxRetryDelay || got <= 0 {
			t.Fatalf("jitteredDelay(%v) = %v, out of bounds", d, got)
		}
	}
	// Jitter must actually vary, otherwise a fleet retries in lockstep.
	seen := map[time.Duration]bool{}
	for i := 0; i < 50; i++ {
		seen[jitteredDelay(maxRetryDelay)] = true
	}
	if len(seen) < 2 {
		t.Fatalf("jitteredDelay produced a single value across 50 draws — no jitter")
	}
}

// End-to-end: a 429 storm must not hang the call. With the pre-fix code this
// test blocks for an hour; with the ceiling it returns promptly.
func TestDoRaw_RateLimitStormDoesNotHang(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "3600")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	}))
	defer srv.Close()

	c, err := NewClient("eg_test_key", WithBaseURL(srv.URL))
	if err != nil {
		t.Fatal(err)
	}

	// context.Background(), not t.Context(): t.Context() is Go 1.24+ and this
	// module declares `go 1.21`, so `go vet ./...` failed the whole package on
	// it ("testing.Context requires go1.24 or later"). That failure was invisible
	// until 2026-08-03 because an earlier compile error in the package masked it.
	// Bumping the go directive would silently drop older toolchains for every
	// consumer of a published module, so the test drops the 1.24-only helper
	// instead — the context is incidental to what this test asserts.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan struct{})
	go func() {
		_, _, _ = c.doRaw(ctx, http.MethodGet, "/health", "", nil)
		close(done)
	}()

	// maxRetries=3 → 2 sleeps, each <= maxRetryDelay (10s). Generous ceiling.
	select {
	case <-done:
	case <-time.After(2*maxRetryDelay + 10*time.Second):
		t.Fatal("doRaw did not return: the SDK is honouring Retry-After unbounded")
	}
}
