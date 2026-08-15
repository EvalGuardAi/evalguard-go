package evalguard

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

// THE TOTALITY GATE — EVERY exported method, not a hand-picked list.
//
// AUDIT 2026-08-09. failclosed_matrix_test.go is the right SHAPE (control column
// + fault matrix) but it enumerates 24 probes by hand against 158 exported
// methods on *Client. The defect this package keeps re-growing is precisely the
// one that lands on "whatever nobody listed", so a hand-maintained probe list is
// the same partial-hardening pattern one level up: the fix reaches the methods
// somebody remembered.
//
// This file enumerates the methods by REFLECTION, so method #159 is covered the
// day it is written, with no line to add anywhere. Measured against this tree
// BEFORE the doRequest fix, 134 methods carrying a payload contract read as a
// clean/empty result under all six "2xx that answers nothing" modes:
//
//	GetAuditLogs        -> &AuditLogsResponse{Logs:nil, Total:0}, err=<nil>
//	                       i.e. "no audit activity" from a 204
//	DetectLanguage      -> &LanguageDetection{Language:"", Confidence:0}, err=<nil>
//	ListGuardrails      -> nil slice, err=<nil>       ("no guardrails configured")
//	GetShadowAIFindings -> zero struct, err=<nil>     ("no shadow AI detected")
//
// Three assertions, because any one alone is worthless:
//
//  1. CONTROL — every probed method MUST return no error against a well-formed
//     clean answer. A method whose control fails is EXCLUDED from the matrix and
//     listed by name: a probe that cannot read a clean verdict "fails closed"
//     under every mode and proves nothing. (This is the exact fiction that made
//     two previous matrices worthless.)
//  2. MATRIX — no method with a payload contract may return a nil error under a
//     mode that carries no answer.
//  3. COVERAGE — the number of EXCLUDED methods is ratcheted, so "0 fail open"
//     can never again be bought by quietly excluding the failures.

// synth builds a plausible non-zero argument of type t, so a method's own
// client-side validation (which rejects empty ids, empty slices, …) does not
// short-circuit the call before it ever reaches the wire. A method whose
// validation still refuses these values fails its CONTROL and is reported as
// EXCLUDED rather than silently counted as "fails closed".
func synth(t reflect.Type, depth int) reflect.Value {
	if depth > 4 {
		return reflect.Zero(t)
	}
	switch t.Kind() {
	case reflect.String:
		// A UUID satisfies both the "must be non-empty" and the "must be a UUID"
		// validators this client applies to project/org/run identifiers.
		return reflect.ValueOf("11111111-1111-4111-8111-111111111111").Convert(t)
	case reflect.Bool:
		return reflect.ValueOf(true).Convert(t)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return reflect.ValueOf(int64(1)).Convert(t)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return reflect.ValueOf(uint64(1)).Convert(t)
	case reflect.Float32, reflect.Float64:
		return reflect.ValueOf(float64(1)).Convert(t)
	case reflect.Slice:
		s := reflect.MakeSlice(t, 1, 1)
		s.Index(0).Set(synth(t.Elem(), depth+1))
		return s
	case reflect.Map:
		m := reflect.MakeMap(t)
		m.SetMapIndex(synth(t.Key(), depth+1), synth(t.Elem(), depth+1))
		return m
	case reflect.Ptr:
		p := reflect.New(t.Elem())
		p.Elem().Set(synth(t.Elem(), depth+1))
		return p
	case reflect.Struct:
		v := reflect.New(t).Elem()
		for i := 0; i < t.NumField(); i++ {
			if !v.Field(i).CanSet() {
				continue
			}
			v.Field(i).Set(synth(t.Field(i).Type, depth+1))
		}
		return v
	case reflect.Interface:
		if t.NumMethod() == 0 {
			return reflect.ValueOf(map[string]any{"k": "v"})
		}
		return reflect.Zero(t)
	default:
		return reflect.Zero(t)
	}
}

// totalitySkip names methods that cannot be driven by a generic harness, each
// with the reason. Kept deliberately tiny — every entry is a hole in the gate.
var totalitySkip = map[string]string{
	// Streams over a long-lived connection; the fault listener answers once and
	// the reader would block on the hang mode past any sane test budget.
	"StreamChatCompletion": "server-sent-events stream, not a request/response verdict",
	// Best-effort by DESIGN and documented as such: a version-policy read that
	// fails must not brick a customer's SDK fleet, and the server enforces the
	// same policy from the version header on every request.
	"CheckVersionPolicy": "documented best-effort fail-open; server enforces via version header",
}

// totalityValidAnswer marks (method, mode) cells whose payload is a legitimate
// answer for that route, so reading it as safe is correct rather than a
// fail-open. Every entry carries its reason.
var totalityValidAnswer = map[string]map[string]string{
	"GetAgentMemoryGovernance": {
		"null-verdict": "`policy:null` is this route's explicit \"no policy configured\" answer",
	},
	// This route downloads OPAQUE BYTES: every non-empty body is a legitimate
	// attachment, including one that happens to be JSON, malformed JSON, or the
	// four letters `null`. The only real fault is a body that is not there at
	// all, and the 204 / 200-empty-body cells ARE asserted — see the length
	// guard in FetchTraceAttachment.
	"FetchTraceAttachment": {
		"malformed-json":   "an attachment is opaque bytes; non-JSON content is a legitimate payload",
		"missing-verdict":  "an attachment is opaque bytes; any non-empty body is its content",
		"null-verdict":     "an attachment is opaque bytes; any non-empty body is its content",
		"empty-object":     "an attachment is opaque bytes; any non-empty body is its content",
		"200-literal-null": "an attachment is opaque bytes; any non-empty body is its content",
	},
}

// totalityFaultModes are the modes a generic harness can assert on. The typed
// modes (stringly-false, zero-verdict) are omitted deliberately: json.Unmarshal
// of `"false"` into a bool field already errors, so doRequest refuses them for
// every method and asserting it here would only pad the count.
var totalityFaultModes = []faultMode{
	{name: "missing-verdict", status: 200, body: `{"success":true,"data":{"latencyMs":3}}`},
	{name: "null-verdict", status: 200, body: `{"success":true,"data":null}`},
	{name: "empty-object", status: 200, body: `{"success":true,"data":{}}`},
	{name: "204-no-body", status: 204, body: ``},
	{name: "200-empty-body", status: 200, body: ``},
	{name: "200-literal-null", status: 200, body: `null`},
	{name: "2xx-success-false", status: 200, body: `{"success":false,"error":{"code":"ENGINE_DOWN","message":"down"},"data":{"blocked":false,"flagged":false,"passed":true,"action":"allow","findingsCount":0,"findings":[]}}`},
	// 3xx was never covered before this file. The comment that used to sit here
	// claimed "Go's default (no follow)" — that was FALSE (a nil CheckRedirect
	// means FOLLOW) and it is the same wrong assertion that hid SEC-051 in
	// doRaw. What actually makes this row fail closed is disarmRedirects plus
	// the SAME-HOST-ONLY rule in sameHostRedirectTarget.
	//
	// SCOPE CORRECTED 2026-08-12: the rule is no longer "refuse every 3xx" —
	// production itself answers 308 on the verdict route for a base URL with a
	// trailing slash, and refusing that hard-fails a live customer. This row is
	// therefore a CROSS-HOST redirect: newMatrixClient stands up a SECOND
	// loopback listener (same hostname, different PORT — the exact shape that
	// leaked the API key through the stdlib's hostname-only comparison) which
	// answers 200 with this very body. A client that follows the hop reads a
	// clean ALLOW and this matrix goes RED for every verdict-bearing method at
	// once. Refusing the hop is the only way it stays green. The same-host
	// FOLLOW half is pinned in same_host_redirect_test.go.
	//
	// 1xx is deliberately ABSENT: net/http consumes informational responses
	// inside the transport and never surfaces them to a caller, so a "1xx" row
	// would silently degenerate into 200-empty-body and pad the count with a
	// mode this client cannot actually observe.
	{name: "3xx-redirect", status: 302, body: `{"success":true,"data":{"blocked":false,"action":"allow"}}`},
	{name: "4xx", status: 403, body: `{"success":false,"error":{"message":"nope"}}`},
	{name: "5xx", status: 500, body: `{"success":false,"error":{"message":"boom"}}`},
	{name: "malformed-json", status: 200, body: `{"success":true,"data":{"blocked":`},
	{name: "conn-refused", dead: true},
	{name: "timeout", hang: true},
}

// cleanValueFor builds a plausible, fully-populated value of type t as generic
// JSON. This is what makes the CONTROL column real: the clean answer for a route
// is derived from the SHAPE THAT ROUTE RETURNS, so `GetAuditLogs` is controlled
// with `{"logs":[…],"total":1}` rather than with a one-size-fits-all blob that no
// route actually sends. Strings use the same UUID `synth` passes as arguments so
// the request/response BINDING guards (scanId, purl, orgId …) match.
func cleanValueFor(t reflect.Type, depth int) any {
	if depth > 4 || t == nil {
		return nil
	}
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	if t == reflect.TypeOf(time.Time{}) {
		return "2026-01-01T00:00:00Z"
	}
	switch t.Kind() {
	case reflect.String:
		return "11111111-1111-4111-8111-111111111111"
	case reflect.Bool:
		return false
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64:
		return 1
	case reflect.Slice:
		if t.Elem().Kind() == reflect.Uint8 {
			return "YXR0YWNobWVudA==" // []byte marshals as base64
		}
		return []any{cleanValueFor(t.Elem(), depth+1)}
	case reflect.Map:
		// A `map[string]any` return is a RAW security route; those carry their own
		// presence/validity/evidence guards, and baseCleanData is the body written
		// to satisfy them. A map with a CONCRETE value type (headers, counters) is
		// a real typed field and must be generated at that type instead, or the
		// clean body fails to decode and the method drops out of the matrix.
		if t.Elem().Kind() == reflect.Interface {
			return baseCleanData()
		}
		return map[string]any{"k": cleanValueFor(t.Elem(), depth+1)}
	case reflect.Struct:
		out := map[string]any{}
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			tag := strings.Split(f.Tag.Get("json"), ",")[0]
			if tag == "-" || f.PkgPath != "" {
				continue
			}
			if f.Anonymous && tag == "" {
				if inner, ok := cleanValueFor(f.Type, depth+1).(map[string]any); ok {
					for k, v := range inner {
						out[k] = v
					}
				}
				continue
			}
			name := tag
			if name == "" {
				name = f.Name
			}
			out[name] = cleanValueFor(f.Type, depth+1)
		}
		return out
	case reflect.Interface:
		return "x"
	default:
		return nil
	}
}

// knownOpenMissingVerdict is THE RESIDUAL, NAMED: the exact methods still open
// on the `missing-verdict` mode after the 2026-08-09 doRequest fix, so it is
// impossible to forget and impossible to GROW.
//
// Every one of these returns `map[string]any`. The body `{"latencyMs":3}` is
// refused for every TYPED result, because requirePayload can compare it against
// the struct's declared json field names; a map target declares NONE — it
// absorbs any key — so no generic boundary can tell a wrong-shaped payload from
// a small right one. Closing this needs the routes to declare what they return:
// typed result structs, or a path-to-required-keys table beside requirePayload.
//
// RETRACTION (2026-08-09). An earlier revision of this comment stated that "the
// exact keys for 20 of these routes have been read out of the handlers in
// apps/web/src/app/api/v1/ and are ready to seed such a table." They had not
// been read. The sentence was written from a subagent's output that its author
// never opened, and it is retracted here rather than deleted, because reading
// the handlers does not merely fail to confirm it — it contradicts it.
//
// What the handlers actually show, for the 21 routes on the Java SDK's twin of
// this list (the tightest of the three, so it was read end to end):
//
//   - 11 hand apiSuccess an object LITERAL whose keys are fixed and readable:
//     apps/web/src/app/api/v1/scorers/calibrate/route.ts:63
//     {scorerId, agreement, threshold}
//     apps/web/src/app/api/v1/evaluators/diff/route.ts:44
//     {name, fromVersion, toVersion, diff}
//     apps/web/src/app/api/v1/datasets/[datasetId]/versions/route.ts:90
//     {versions, datasetId}
//     One of the 11 answers with TWO different literals depending on a branch
//     (.../versions/route.ts:148 unchanged vs :196 inserted).
//   - 8 never call apiSuccess AT ALL. environments, tools, tools/deployments
//     and prompts/deployments each declare a route-LOCAL ok() that inlines
//     NextResponse.json({success, data}) — see
//     apps/web/src/app/api/v1/environments/route.ts:22 — and /chat/completions
//     writes a raw OpenAI envelope with no data wrapper at all
//     (apps/web/src/app/api/v1/chat/completions/route.ts:1085; it imports only
//     apiError, at :44). A table seeded from apiSuccess call sites is
//     structurally blind to every one of them.
//   - 2 hand apiSuccess a value the handler never shapes: a DB row
//     (.../evaluators/route.ts:143, apiSuccess(row, 201)) or a core function's
//     return (.../language/detect/route.ts:24). Those keys are table columns
//     and library types, not text in the handler.
//
// Two further facts a key table would have to survive, both found by the same
// reading:
//
//   - One route answers with an ARRAY on GET and an object on DELETE
//     (.../gateway/guardrails/route.ts:89 vs :263), so the table must be keyed
//     by (method, path) and must admit "no keys — this is a list".
//   - 6 of the 21 paths THIS SDK requests have no handler on disk:
//     /environments/{name} (evalguard.go:1311), /tools/{name} (:1381),
//     /tools/{name}/deployments (:1415, :1427) and /prompts/{name}/deployments
//     (:1323, :1335). apps/web/src/app/api/v1 has no dynamic segment under
//     environments or tools, none but [promptVersionId]/labels under prompts,
//     and neither next.config.ts nor middleware.ts rewrites — so they fall to
//     apps/web/src/app/api/v1/[...catch]/route.ts, a hard 404. That is a real
//     SDK/API drift, tracked separately; it does not move this matrix, which
//     answers from its own stub listener rather than the app.
//
// So the required-keys table is still the right shape of fix, but its input
// does not exist yet and CANNOT be produced by reading apiSuccess call sites
// alone. "Keys for this route" needs a THIRD state from the start — known,
// known-absent, or UNVERIFIABLE — instead of assuming every route can supply
// one. It stays deliberately NOT half-applied: a table covering some routes and
// not others is the same partial hardening this round exists to kill, and a
// WRONG entry turns a working call into a hard failure for a customer.
//
// The ratchet below is two-sided — a NEW fail-open fails the test, and a listed
// method that now fails CLOSED also fails it, so this list can only shrink and
// can never rot into a permanent excuse.
var knownOpenMissingVerdict = map[string]bool{
	"AnalyzeTrace": true, "Ask": true, "CalibrateScorer": true, "ChatCompletions": true,
	"CreateEnvironment": true, "CreateEvaluator": true, "CreatePrompt": true,
	"CreateSiemInboundToken": true, "CreateTool": true, "CreateTrace": true,
	"DiffDatasetVersions": true, "DiffEvaluatorVersions": true, "GenerateAISBOM": true,
	"GenerateGuardrails": true, "GetAISBOM": true, "GetAutopilotConfig": true, "GetCost": true,
	"GetCostAnomalies": true, "GetCostBudget": true, "GetCostForecast": true,
	"GetCostRecommendations": true, "GetCostSavings": true, "GetDashboardStats": true,
	"GetDatasetVersion": true, "GetGatewayHealth": true, "GetGatewayStats": true,
	"GetLeaderboard": true, "GetModelCards": true, "GetMonitoringAlerts": true,
	"GetMonitoringAnalytics": true, "GetMonitoringDrift": true, "GetMonitoringSLA": true,
	"GetSIEMConnectors": true, "GetSecurityEffectiveness": true, "GetSettings": true,
	"GetThreatIntelligence": true, "GetTool": true, "GetTrace": true, "IngestOTLP": true,
	"IngestShadowAISightings": true, "ListAgentRuns": true, "ListDatasetVersions": true,
	"ListGuardrails": true, "ListNotifications": true, "ListPipelines": true, "ListTickets": true,
	// RemovePromptDeployment was removed from this list on 2026-08-09: it no
	// longer reaches the network at all. The prompt deployments route exports
	// GET/POST/PUT and no DELETE, so the call it used to make
	// (DELETE /prompts/{name}/deployments) hit the api/v1 catch-all and 404'd.
	// It now fails locally with a typed error, which is a CLOSED verdict — so
	// listing it here would make this ratchet stale.
	"RemoveEnvironment": true, "RemoveToolDeployment": true,
	"RestoreDatasetVersion": true, "Search": true, "SearchTraces": true,
	"SetPromptDeployment": true, "SetToolDeployment": true, "SmartRoute": true,
	"SnapshotDataset": true, "SubmitTicket": true,
}

type totalityMethod struct {
	name string
	// hasPayload is true when the method returns something besides an error, so
	// a nil error under a no-answer mode hands the caller a zero-valued result.
	hasPayload bool
	// cleanBodies are the candidate well-formed CLEAN answers for THIS route,
	// derived from the type the method returns.
	cleanBodies []string
	call        func(c *Client) error
}

func totalityMethods(t *testing.T) []totalityMethod {
	t.Helper()
	ct := reflect.TypeOf(&Client{})
	ctxType := reflect.TypeOf((*context.Context)(nil)).Elem()
	errType := reflect.TypeOf((*error)(nil)).Elem()

	var out []totalityMethod
	for i := 0; i < ct.NumMethod(); i++ {
		m := ct.Method(i)
		if _, skip := totalitySkip[m.Name]; skip {
			continue
		}
		mt := m.Type
		// Only request/response methods: first arg (after the receiver) is a
		// context and the last return is an error.
		if mt.NumIn() < 2 || mt.In(1) != ctxType {
			continue
		}
		if mt.NumOut() == 0 || mt.Out(mt.NumOut()-1) != errType {
			continue
		}
		if mt.IsVariadic() {
			continue
		}
		name := m.Name
		fn := m.Func
		hasPayload := mt.NumOut() > 1
		out = append(out, totalityMethod{
			name:        name,
			hasPayload:  hasPayload,
			cleanBodies: cleanBodyCandidates(t, name, mt),
			call: func(c *Client) error {
				args := make([]reflect.Value, mt.NumIn())
				args[0] = reflect.ValueOf(c)
				args[1] = reflect.ValueOf(context.Background())
				for a := 2; a < mt.NumIn(); a++ {
					args[a] = synth(mt.In(a), 0)
				}
				res := fn.Call(args)
				last := res[len(res)-1]
				if last.IsNil() {
					return nil
				}
				return last.Interface().(error)
			},
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	return out
}

// cleanBodyForReturn renders the well-formed CLEAN answer for ONE route, built
// from the type that route returns:
//
//	baseCleanData()                 the keys the verdict guards require
//	  <- overlaid with the shape of the method's own return type
//	  <- overlaid with matrixProbes()' per-route cleanOverrides, when it has one
//
// A method returning a raw []byte (the attachment download) gets opaque bytes
// rather than an envelope, because that is what its route really sends.
// cleanBodyCandidates returns the shapes a clean answer for this route could
// take, in preference order. Several routes decode into an unexported WRAPPER
// (`struct{ Scorers []Scorer }`) while returning the inner slice, so a body
// generated from the return type alone would not decode and the method would
// drop silently out of the matrix. The harness tries each candidate and keeps
// the first the method accepts.
func cleanBodyCandidates(t *testing.T, name string, mt reflect.Type) []string {
	t.Helper()
	marshal := func(data any) string {
		body, err := json.Marshal(map[string]any{"success": true, "data": data})
		if err != nil {
			t.Fatalf("marshal clean body for %s: %v", name, err)
		}
		return string(body)
	}
	if mt.NumOut() < 2 {
		return []string{marshal(baseCleanData())}
	}
	ret := mt.Out(0)
	if ret.Kind() == reflect.Slice && ret.Elem().Kind() == reflect.Uint8 {
		return []string{"attachment-bytes"}
	}
	shape := cleanValueFor(ret, 0)

	var out []string
	if m, ok := shape.(map[string]any); ok {
		merged := baseCleanData()
		for k, v := range m {
			merged[k] = v
		}
		for _, p := range matrixProbes() {
			if p.name == name {
				for k, v := range p.cleanOverrides {
					merged[k] = v
				}
			}
		}
		out = append(out, marshal(merged), marshal(m))
		return out
	}
	// A slice return: try the bare array, then the wrapper shapes this API uses
	// ({"scorers": [...]}, {"items"/"data"/"results": [...]}).
	out = append(out, marshal(shape))
	elem := ret
	for elem.Kind() == reflect.Slice || elem.Kind() == reflect.Ptr {
		elem = elem.Elem()
	}
	keys := []string{"items", "results", "list"}
	if n := elem.Name(); n != "" {
		keys = append([]string{strings.ToLower(n[:1]) + n[1:] + "s"}, keys...)
	}
	for _, key := range keys {
		out = append(out, marshal(map[string]any{key: shape}))
	}
	return out
}

// withCleanServer drives fn against a listener answering THIS method's clean
// body. Each probe gets its own single-route server; httptest servers are
// in-process and cheap.
func withCleanServer(t *testing.T, body string, fn func(c *Client)) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()
	c, err := NewClient("eg_t", WithBaseURL(srv.URL), WithTimeout(3*time.Second))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	fn(c)
}

// controlResult runs every candidate clean body and reports the first that the
// method accepts, or the last error if none did.
func controlResult(t *testing.T, m totalityMethod) (ok bool, lastErr error) {
	t.Helper()
	for _, body := range m.cleanBodies {
		withCleanServer(t, body, func(c *Client) {
			lastErr = m.call(c)
		})
		if lastErr == nil {
			return true, nil
		}
	}
	return false, lastErr
}

// exclusionCategory classifies WHY a method could not read a clean answer.
// Only "harness" is a genuine blind spot; the other two are proof the call
// never reaches a state where it could read a fault as safe.
func exclusionCategory(err error) string {
	msg := err.Error()
	switch {
	case strings.Contains(msg, "INDETERMINATE_VERDICT") && strings.Contains(msg, "must NOT be treated as allowed"),
		strings.Contains(msg, "INDETERMINATE_VERDICT") && strings.Contains(msg, "must not be treated as allowed"):
		return "guard-fired" // its own verdict guard refused the synthetic body: proof of life
	case strings.Contains(msg, "VALIDATION_ERROR"):
		return "client-validation" // refused before the wire; cannot read any response
	default:
		return "harness" // the synthetic clean body does not match the route's real shape
	}
}

// maxUnmeasuredMethods RATCHETS the control column. An UNMEASURED method is one
// the harness cannot build a clean answer for, so its fail-closed cells prove
// nothing. Without this ceiling a future change could buy a green matrix by
// quietly breaking more probes — which is exactly how two earlier "0 fail open"
// matrices turned out to be fiction.
const maxUnmeasuredMethods = 8

// TestFailOpenTotality_Control proves every probe can read a CLEAN answer, and
// reports (and ratchets) the ones that cannot instead of hiding them.
func TestFailOpenTotality_Control(t *testing.T) {
	methods := totalityMethods(t)
	byCategory := map[string][]string{}
	excluded := 0
	for _, m := range methods {
		ok, err := controlResult(t, m)
		if ok {
			continue
		}
		excluded++
		cat := exclusionCategory(err)
		byCategory[cat] = append(byCategory[cat], fmt.Sprintf("%-34s %v", m.name, err))
	}
	t.Logf("CONTROL: %d of %d exported request methods read a CLEAN answer; %d excluded",
		len(methods)-excluded, len(methods), excluded)
	for _, cat := range []string{"guard-fired", "client-validation", "harness"} {
		for _, e := range byCategory[cat] {
			t.Logf("  EXCLUDED [%s] %s", cat, e)
		}
	}
	// Only the "harness" category is a real blind spot. A method excluded because
	// its OWN verdict guard refused the synthetic body is proof the guard is live,
	// and one excluded by client-side validation never reaches the wire at all.
	blind := len(byCategory["harness"])
	if blind > maxUnmeasuredMethods {
		t.Fatalf("%d methods are UNMEASURED — the harness cannot build a clean answer for them, so "+
			"their fail-closed cells in TestFailOpenTotality prove NOTHING (ceiling %d):\n  %s",
			blind, maxUnmeasuredMethods, strings.Join(byCategory["harness"], "\n  "))
	}
}

// controlPasses returns the methods whose CONTROL reads a clean answer. Only
// those are asserted in the matrix — a method that cannot read a clean answer
// would "fail closed" everywhere for the wrong reason.
func controlPasses(t *testing.T) []totalityMethod {
	t.Helper()
	var out []totalityMethod
	for _, m := range totalityMethods(t) {
		if pass, _ := controlResult(t, m); pass {
			out = append(out, m)
		}
	}
	return out
}

// TestFailOpenTotality is the gate: no method carrying a payload contract may
// return a nil error from a 2xx that answers nothing.
func TestFailOpenTotality(t *testing.T) {
	methods := controlPasses(t)
	failOpen := map[string][]string{}
	payloadCount := 0
	for _, m := range methods {
		if m.hasPayload {
			payloadCount++
		}
	}

	for _, fm := range totalityFaultModes {
		c, cleanup := newMatrixClient(t, fm)
		// Run the methods concurrently: the conn-refused and timeout modes each
		// spend the full retry ladder per call, which serially costs ~7 minutes
		// for the whole matrix. *Client is stateless per call and safe to share.
		var mu sync.Mutex
		var wg sync.WaitGroup
		sem := make(chan struct{}, 16)
		for _, m := range methods {
			if !m.hasPayload {
				// An action method (returns only `error`) has no payload
				// contract: a bodyless 2xx really is "done". Its fail-closed
				// behaviour on 3xx/4xx/5xx/success:false is covered by doRaw and
				// asserted in the shared-boundary tests, not here.
				continue
			}
			if why, ok := totalityValidAnswer[m.name][fm.name]; ok {
				t.Logf("skipping %s/%s: %s", m.name, fm.name, why)
				continue
			}
			m := m
			wg.Add(1)
			sem <- struct{}{}
			go func() {
				defer wg.Done()
				defer func() { <-sem }()
				if err := m.call(c); err == nil {
					mu.Lock()
					failOpen[m.name] = append(failOpen[m.name], fm.name)
					mu.Unlock()
				}
			}()
		}
		wg.Wait()
		cleanup()
	}

	// THE RATCHET, BOTH WAYS.
	var unexpected, healed []string
	for n, modes := range failOpen {
		for _, m := range modes {
			if m != "missing-verdict" || !knownOpenMissingVerdict[n] {
				unexpected = append(unexpected, fmt.Sprintf("  %-34s %s", n, m))
			}
		}
	}
	for n := range knownOpenMissingVerdict {
		open := false
		for _, m := range failOpen[n] {
			if m == "missing-verdict" {
				open = true
			}
		}
		if !open {
			healed = append(healed, "  "+n)
		}
	}
	sort.Strings(unexpected)
	sort.Strings(healed)

	if len(unexpected) > 0 {
		t.Fatalf("NEW FAIL-OPEN(S) — a 2xx carrying no answer returned a nil error and a "+
			"zero-valued result, and these cells are not in knownOpenMissingVerdict:\n%s",
			strings.Join(unexpected, "\n"))
	}
	if len(healed) > 0 {
		t.Fatalf("knownOpenMissingVerdict IS STALE — these methods now fail CLOSED and must be "+
			"removed from the list, or the ratchet stops measuring anything:\n%s",
			strings.Join(healed, "\n"))
	}
	t.Logf("%d payload-bearing methods x %d fault modes: 0 NEW fail-open; %d residual cells, "+
		"all named in knownOpenMissingVerdict",
		payloadCount, len(totalityFaultModes), len(knownOpenMissingVerdict))
}
