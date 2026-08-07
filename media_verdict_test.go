package evalguard

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// ─────────────────────────────────────────────────────────────────────────────
// CLASS 1, THIRD PASS — the multimodal moderation verdicts.
//
// verdict_presence_test.go pins the typed-struct half of this class (MCP audit,
// agent-exec red-team, RAG scan, voice deepfake) and firewall_verdict_test.go
// pins /firewall/check. This file pins the three methods that returned
// map[string]any, which is the WORST shape of the defect: the caller's gate is
// `res["flagged"].(bool)` / `res["synthetic"].(bool)`, and a type assertion on an
// ABSENT key yields the zero value — false — without panicking the way a WRONG
// type does.
//
//	ModerateImage        → `flagged` absent ⇒ false, `score` absent ⇒ 0.0 on a
//	                       0..1 harm scale: "clean image, zero harm".
//	ModerateVideo        → same, plus framesEvaluated 0: "clean clip" and "no
//	                       frame was ever moderated" were the same answer.
//	DetectMediaDeepfake  → `synthetic` absent ⇒ false AND `probability` absent
//	                       ⇒ 0.0, so BOTH natural gates read "authentic".
//
// Measured on this tree before the fix, against the loopback server below: every
// body in mediaNoVerdictBodies returned err=nil from all three methods.
// ─────────────────────────────────────────────────────────────────────────────

// mediaNoVerdictBodies is the full set of 2xx shapes that carry no verdict. Each
// one is a real thing a proxy, a gateway, a truncated response, or schema drift
// puts on the wire with a 200.
var mediaNoVerdictBodies = []struct {
	name string
	body string
}{
	{"empty object", `{}`},
	{"empty body", ``},
	{"bare null", `null`},
	{"null data", `{"success":true,"data":null}`},
	{"envelope with no verdict", `{"success":true,"data":{"provider":"openai","latencyMs":184}}`},
	{"unrelated 200", `{"success":true,"data":{"id":"evt_9f2","object":"event","created":1754400000}}`},
	{"proxy error envelope on a 200", `{"success":true,"data":{"error":"upstream timeout"}}`},
	{"explicit nulls for the decision fields",
		`{"success":true,"data":{"flagged":null,"score":null,"synthetic":null,"probability":null,` +
			`"frames":null,"meanProbability":null}}`},
	{"empty data object", `{"success":true,"data":{}}`},
	{"bare array", `[]`},
	{"json string", `"ok"`},
}

// assertNoReadableMap fails when a refused call still handed back a map. The
// refusal is only worth anything if the zero-valued map is UNREACHABLE — leaving
// it readable puts the zero value back in front of the next call site.
func assertNoReadableMap(t *testing.T, label string, res map[string]any) {
	t.Helper()
	if res != nil {
		t.Errorf("%s: must not hand back a readable map alongside the refusal; got %+v", label, res)
	}
}

func TestModerateImage_NoVerdictMustNotReadAsClean(t *testing.T) {
	for _, tc := range mediaNoVerdictBodies {
		t.Run(tc.name, func(t *testing.T) {
			c, cleanup := verdictServer(t, tc.body)
			defer cleanup()
			res, err := c.ModerateImage(context.Background(), &ModerateImageRequest{
				OrgID: "org", ProjectID: "proj", ImageURL: "https://example.com/a.png",
			})
			assertIndeterminate(t, tc.name, err, res)
			assertNoReadableMap(t, tc.name, res)
		})
	}
}

func TestModerateVideo_NoVerdictMustNotReadAsClean(t *testing.T) {
	for _, tc := range mediaNoVerdictBodies {
		t.Run(tc.name, func(t *testing.T) {
			c, cleanup := verdictServer(t, tc.body)
			defer cleanup()
			res, err := c.ModerateVideo(context.Background(), &ModerateVideoRequest{
				OrgID: "org", ProjectID: "proj",
				Frames: []ModerationFrame{{ImageURL: "https://example.com/f0.png"}},
			})
			assertIndeterminate(t, tc.name, err, res)
			assertNoReadableMap(t, tc.name, res)
		})
	}
}

func TestDetectMediaDeepfake_NoVerdictMustNotReadAsAuthentic(t *testing.T) {
	for _, tc := range mediaNoVerdictBodies {
		t.Run("image/"+tc.name, func(t *testing.T) {
			c, cleanup := verdictServer(t, tc.body)
			defer cleanup()
			res, err := c.DetectMediaDeepfake(context.Background(), &DetectMediaDeepfakeRequest{
				OrgID: "org", ProjectID: "proj", ImageURL: "https://example.com/a.png",
			})
			assertIndeterminate(t, tc.name, err, res)
			assertNoReadableMap(t, tc.name, res)
		})
		t.Run("video/"+tc.name, func(t *testing.T) {
			c, cleanup := verdictServer(t, tc.body)
			defer cleanup()
			res, err := c.DetectMediaDeepfake(context.Background(), &DetectMediaDeepfakeRequest{
				OrgID: "org", ProjectID: "proj",
				Frames: []ModerationFrame{{ImageURL: "https://example.com/f0.png"}},
			})
			assertIndeterminate(t, tc.name, err, res)
			assertNoReadableMap(t, tc.name, res)
		})
	}
}

// ─── the other half: an EXPLICIT verdict must still parse ────────────────────
//
// A refusal that also refuses healthy traffic is not a fix, it is an outage. The
// bodies below are what the real routes emit (apiSuccess(moderateImage(...)) /
// apiSuccess({kind, ...detect...})), including the benign ones — an explicit
// `flagged:false` is a REAL allow and must survive.

const (
	imageGenuineClean = `{"success":true,"data":{"flagged":false,"score":0.021,"categories":[],` +
		`"categoryScores":{"violence":0.021,"sexual":0.004},"provider":"openai","latencyMs":184}}`
	imageGenuineFlagged = `{"success":true,"data":{"flagged":true,"score":0.934,"categories":["violence"],` +
		`"categoryScores":{"violence":0.934},"provider":"openai","latencyMs":201}}`

	videoGenuineClean = `{"success":true,"data":{"flagged":false,"score":0.1,"categories":[],` +
		`"framesTotal":3,"framesEvaluated":3,"frames":[` +
		`{"index":0,"timestampMs":0,"flagged":false,"score":0.1,"categories":[]},` +
		`{"index":1,"timestampMs":500,"flagged":false,"score":0.05,"categories":[]},` +
		`{"index":2,"timestampMs":1000,"flagged":false,"score":0.02,"categories":[]}],` +
		`"provider":"openai","latencyMs":903}}`
	videoGenuineFlagged = `{"success":true,"data":{"flagged":true,"score":0.91,"categories":["sexual"],` +
		`"firstFlaggedFrame":1,"framesTotal":3,"framesEvaluated":3,"frames":[` +
		`{"index":0,"flagged":false,"score":0.1,"categories":[]},` +
		`{"index":1,"flagged":true,"score":0.91,"categories":["sexual"]},` +
		`{"index":2,"flagged":false,"score":0.2,"categories":[]}],` +
		`"provider":"openai","latencyMs":911}}`

	deepfakeGenuineAuthentic = `{"success":true,"data":{"kind":"image","synthetic":false,"probability":0.08,` +
		`"label":"real","scores":[{"label":"real","score":0.92}],"provider":"sidecar","latencyMs":120}}`
	deepfakeGenuineSynthetic = `{"success":true,"data":{"kind":"image","synthetic":true,"probability":0.97,` +
		`"label":"fake","scores":[{"label":"fake","score":0.97}],"provider":"sidecar","latencyMs":131}}`
	deepfakeVideoGenuine = `{"success":true,"data":{"kind":"video","synthetic":true,"probability":0.9,` +
		`"meanProbability":0.5,"firstSyntheticFrame":1,"framesTotal":2,"framesEvaluated":2,"frames":[` +
		`{"index":0,"timestampMs":0,"synthetic":false,"probability":0.1},` +
		`{"index":1,"timestampMs":40,"synthetic":true,"probability":0.9}],` +
		`"provider":"sidecar","latencyMs":420}}`
)

func TestModerateImage_RealVerdictsStillParse(t *testing.T) {
	for _, tc := range []struct {
		name        string
		body        string
		wantFlagged bool
		wantScore   float64
	}{
		{"explicit allow", imageGenuineClean, false, 0.021},
		{"explicit block", imageGenuineFlagged, true, 0.934},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, cleanup := verdictServer(t, tc.body)
			defer cleanup()
			res, err := c.ModerateImage(context.Background(), &ModerateImageRequest{
				OrgID: "org", ProjectID: "proj", ImageBase64: "aGk=",
			})
			if err != nil {
				t.Fatalf("unexpected error on a real verdict: %v", err)
			}
			if got, ok := res["flagged"].(bool); !ok || got != tc.wantFlagged {
				t.Errorf("flagged: want %t, got %v (present=%t)", tc.wantFlagged, res["flagged"], ok)
			}
			if got, ok := res["score"].(float64); !ok || got != tc.wantScore {
				t.Errorf("score: want %v, got %v", tc.wantScore, res["score"])
			}
			// The whole body must survive — the guard decodes a second copy, it
			// does not filter the one the caller gets.
			if res["categoryScores"] == nil || res["provider"] != "openai" {
				t.Errorf("non-verdict fields lost: %+v", res)
			}
		})
	}
}

func TestModerateVideo_RealVerdictsStillParse(t *testing.T) {
	for _, tc := range []struct {
		name        string
		body        string
		wantFlagged bool
	}{
		{"explicit allow", videoGenuineClean, false},
		{"explicit block", videoGenuineFlagged, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, cleanup := verdictServer(t, tc.body)
			defer cleanup()
			res, err := c.ModerateVideo(context.Background(), &ModerateVideoRequest{
				OrgID: "org", ProjectID: "proj", Frames: threeFrames(),
			})
			if err != nil {
				t.Fatalf("unexpected error on a real verdict: %v", err)
			}
			if got, ok := res["flagged"].(bool); !ok || got != tc.wantFlagged {
				t.Errorf("flagged: want %t, got %v", tc.wantFlagged, res["flagged"])
			}
			if frames, ok := res["frames"].([]any); !ok || len(frames) != 3 {
				t.Errorf("per-frame evidence lost: %+v", res["frames"])
			}
		})
	}
}

func TestDetectMediaDeepfake_RealVerdictsStillParse(t *testing.T) {
	for _, tc := range []struct {
		name          string
		body          string
		req           *DetectMediaDeepfakeRequest
		wantSynthetic bool
	}{
		{"image authentic", deepfakeGenuineAuthentic,
			&DetectMediaDeepfakeRequest{OrgID: "org", ProjectID: "proj", ImageBase64: "aGk="}, false},
		{"image synthetic", deepfakeGenuineSynthetic,
			&DetectMediaDeepfakeRequest{OrgID: "org", ProjectID: "proj", ImageBase64: "aGk="}, true},
		{"video synthetic", deepfakeVideoGenuine,
			&DetectMediaDeepfakeRequest{OrgID: "org", ProjectID: "proj", Frames: twoFrames()}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, cleanup := verdictServer(t, tc.body)
			defer cleanup()
			res, err := c.DetectMediaDeepfake(context.Background(), tc.req)
			if err != nil {
				t.Fatalf("unexpected error on a real verdict: %v", err)
			}
			if got, ok := res["synthetic"].(bool); !ok || got != tc.wantSynthetic {
				t.Errorf("synthetic: want %t, got %v", tc.wantSynthetic, res["synthetic"])
			}
		})
	}
}

// ─── the second half of the class: a verdict PRESENT but unusable ────────────
//
// The refusals above close "no decision field at all". These close "the field is
// there, so a presence check passes, but the value cannot be acted on" — the
// same one-word edit that turned an MCP "block" into "pass", applied to media.

func TestModerateImage_VerdictContradictingItsOwnScoreIsRefused(t *testing.T) {
	// The engine derives `flagged = backend.flagged || score >= threshold`, so a
	// 0.95 harm score with flagged:false is a body it could not have produced.
	c, cleanup := verdictServer(t,
		`{"success":true,"data":{"flagged":false,"score":0.95,"categories":["violence"],"provider":"openai"}}`)
	defer cleanup()
	res, err := c.ModerateImage(context.Background(), &ModerateImageRequest{
		OrgID: "org", ProjectID: "proj", ImageBase64: "aGk=",
	})
	assertIndeterminate(t, "flagged:false at score 0.95", err, res)
	assertNoReadableMap(t, "flagged:false at score 0.95", res)
}

func TestModerateImage_CallerThresholdIsHonoured(t *testing.T) {
	// score 0.4 is clean at the 0.7 default but flagged at a strict 0.3, so the
	// SAME body is coherent for one caller and a contradiction for the other.
	// The threshold has to come from the REQUEST — a response that restated its
	// own threshold could always claim the lenient one.
	body := `{"success":true,"data":{"flagged":false,"score":0.4,"categories":[],"provider":"openai"}}`

	c, cleanup := verdictServer(t, body)
	defer cleanup()
	if _, err := c.ModerateImage(context.Background(), &ModerateImageRequest{
		OrgID: "org", ProjectID: "proj", ImageBase64: "aGk=",
	}); err != nil {
		t.Fatalf("score 0.4 is a legitimate allow at the 0.7 default: %v", err)
	}

	res, err := c.ModerateImage(context.Background(), &ModerateImageRequest{
		OrgID: "org", ProjectID: "proj", ImageBase64: "aGk=", Threshold: 0.3,
	})
	assertIndeterminate(t, "flagged:false at score 0.4 with threshold 0.3", err, res)
}

func TestModerateVideo_ClipVerdictMustAgreeWithItsFrames(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
	}{
		{"clip says clean while a frame is flagged",
			`{"success":true,"data":{"flagged":false,"score":0.91,"categories":[],"framesTotal":3,` +
				`"framesEvaluated":3,"frames":[{"index":0,"flagged":false,"score":0.1,"categories":[]},` +
				`{"index":1,"flagged":true,"score":0.91,"categories":["sexual"]},` +
				`{"index":2,"flagged":false,"score":0.2,"categories":[]}]}}`},
		{"clip score is not the worst frame",
			`{"success":true,"data":{"flagged":false,"score":0.05,"categories":[],"framesTotal":3,` +
				`"framesEvaluated":3,"frames":[{"index":0,"flagged":false,"score":0.1,"categories":[]},` +
				`{"index":1,"flagged":false,"score":0.05,"categories":[]},` +
				`{"index":2,"flagged":false,"score":0.02,"categories":[]}]}}`},
		{"only one of the three submitted frames was moderated",
			`{"success":true,"data":{"flagged":false,"score":0.1,"categories":[],"framesTotal":1,` +
				`"framesEvaluated":1,"frames":[{"index":0,"flagged":false,"score":0.1,"categories":[]}]}}`},
		{"verdict with the per-frame evidence stripped",
			`{"success":true,"data":{"flagged":false,"score":0.1,"categories":[],"framesTotal":3,` +
				`"framesEvaluated":3,"frames":[]}}`},
		{"no frame was ever moderated",
			`{"success":true,"data":{"flagged":false,"score":0,"categories":[],"framesTotal":3,` +
				`"framesEvaluated":0,"frames":[]}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, cleanup := verdictServer(t, tc.body)
			defer cleanup()
			res, err := c.ModerateVideo(context.Background(), &ModerateVideoRequest{
				OrgID: "org", ProjectID: "proj", Frames: threeFrames(),
			})
			assertIndeterminate(t, tc.name, err, res)
			assertNoReadableMap(t, tc.name, res)
		})
	}
}

func TestDetectMediaDeepfake_UnusableVerdictsAreRefused(t *testing.T) {
	image := func() *DetectMediaDeepfakeRequest {
		return &DetectMediaDeepfakeRequest{OrgID: "org", ProjectID: "proj", ImageBase64: "aGk="}
	}
	video := func() *DetectMediaDeepfakeRequest {
		return &DetectMediaDeepfakeRequest{OrgID: "org", ProjectID: "proj", Frames: twoFrames()}
	}
	for _, tc := range []struct {
		name string
		body string
		req  *DetectMediaDeepfakeRequest
	}{
		{"authentic at a 0.97 synthetic probability",
			`{"success":true,"data":{"kind":"image","synthetic":false,"probability":0.97,"provider":"sidecar"}}`,
			image()},
		{"unrecognised kind",
			`{"success":true,"data":{"kind":"audio","synthetic":false,"probability":0.02}}`, image()},
		{"kind absent",
			`{"success":true,"data":{"synthetic":false,"probability":0.02}}`, image()},
		{"an image result answering a clip request",
			`{"success":true,"data":{"kind":"image","synthetic":false,"probability":0.02}}`, video()},
		{"probability outside 0..1",
			`{"success":true,"data":{"kind":"image","synthetic":false,"probability":-1}}`, image()},
		{"clip says authentic while a frame is synthetic",
			`{"success":true,"data":{"kind":"video","synthetic":false,"probability":0.9,"meanProbability":0.5,` +
				`"framesTotal":2,"framesEvaluated":2,"frames":[{"index":0,"synthetic":false,"probability":0.1},` +
				`{"index":1,"synthetic":true,"probability":0.9}]}}`, video()},
		{"clip verdict with the frames stripped",
			`{"success":true,"data":{"kind":"video","synthetic":false,"probability":0.02,"meanProbability":0.02,` +
				`"framesTotal":2,"framesEvaluated":2,"frames":[]}}`, video()},
		{"clip probability is not the worst frame",
			`{"success":true,"data":{"kind":"video","synthetic":false,"probability":0.05,"meanProbability":0.1,` +
				`"framesTotal":2,"framesEvaluated":2,"frames":[{"index":0,"synthetic":false,"probability":0.1},` +
				`{"index":1,"synthetic":false,"probability":0.1}]}}`, video()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, cleanup := verdictServer(t, tc.body)
			defer cleanup()
			res, err := c.DetectMediaDeepfake(context.Background(), tc.req)
			assertIndeterminate(t, tc.name, err, res)
			assertNoReadableMap(t, tc.name, res)
		})
	}
}

// ─── HasVerdict for a caller decoding a stored / proxied body itself ─────────

func TestMediaHasVerdict_OnStoredBodies(t *testing.T) {
	type hasVerdict interface{ HasVerdict() bool }
	for _, tc := range []struct {
		name string
		body string
		into func() hasVerdict
		want bool
	}{
		{"image: no verdict", `{}`, func() hasVerdict { return &ImageModerationResult{} }, false},
		{"image: explicit allow", `{"flagged":false,"score":0.02,"categories":[]}`,
			func() hasVerdict { return &ImageModerationResult{} }, true},
		{"image: explicit block", `{"flagged":true,"score":0.93,"categories":["violence"]}`,
			func() hasVerdict { return &ImageModerationResult{} }, true},
		{"video: no verdict", `{"framesTotal":3}`,
			func() hasVerdict { return &VideoModerationResult{} }, false},
		{"video: coherent clip", `{"flagged":true,"score":0.91,"categories":["sexual"],` +
			`"firstFlaggedFrame":0,"framesTotal":1,"framesEvaluated":1,` +
			`"frames":[{"index":0,"flagged":true,"score":0.91,"categories":["sexual"]}]}`,
			func() hasVerdict { return &VideoModerationResult{} }, true},
		{"video: clip contradicting its frames", `{"flagged":false,"score":0.91,"categories":[],` +
			`"framesTotal":1,"framesEvaluated":1,` +
			`"frames":[{"index":0,"flagged":true,"score":0.91,"categories":["sexual"]}]}`,
			func() hasVerdict { return &VideoModerationResult{} }, false},
		{"deepfake: no verdict", `{"kind":"image"}`,
			func() hasVerdict { return &MediaDeepfakeResult{} }, false},
		{"deepfake: explicit authentic", `{"kind":"image","synthetic":false,"probability":0.08}`,
			func() hasVerdict { return &MediaDeepfakeResult{} }, true},
		{"deepfake: unrecognised kind", `{"kind":"audio","synthetic":false,"probability":0.08}`,
			func() hasVerdict { return &MediaDeepfakeResult{} }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			v := tc.into()
			if err := json.Unmarshal([]byte(tc.body), v); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if got := v.HasVerdict(); got != tc.want {
				t.Errorf("HasVerdict() = %t, want %t for %s", got, tc.want, tc.body)
			}
		})
	}
}

// TestMediaRefusalNamesTheMissingField keeps the operator-facing half honest:
// the message has to say WHICH field was missing so schema drift is
// distinguishable from an outage at a glance.
func TestMediaRefusalNamesTheMissingField(t *testing.T) {
	c, cleanup := verdictServer(t, `{"success":true,"data":{"provider":"openai"}}`)
	defer cleanup()
	_, err := c.ModerateImage(context.Background(), &ModerateImageRequest{
		OrgID: "org", ProjectID: "proj", ImageBase64: "aGk=",
	})
	if err == nil {
		t.Fatal("FAIL-OPEN: no error")
	}
	for _, want := range []string{"ModerateImage", "flagged", "/moderation/image", "must not be treated as allowed"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal message must mention %q; got %q", want, err.Error())
		}
	}
}

func threeFrames() []ModerationFrame {
	return []ModerationFrame{
		{ImageURL: "https://example.com/f0.png", TimestampMs: 0},
		{ImageURL: "https://example.com/f1.png", TimestampMs: 500},
		{ImageURL: "https://example.com/f2.png", TimestampMs: 1000},
	}
}

func twoFrames() []ModerationFrame {
	return []ModerationFrame{
		{ImageURL: "https://example.com/f0.png", TimestampMs: 0},
		{ImageURL: "https://example.com/f1.png", TimestampMs: 40},
	}
}
