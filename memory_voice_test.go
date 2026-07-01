package evalguard

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

// Table tests for the agent-memory + voice-ML SDK methods (2026-06-12). Reuses
// newRecordingServer / recordingServer from evalguard_test.go (same package).

func TestMemoryMethods(t *testing.T) {
	ctx := context.Background()
	pid := "11111111-1111-4111-8111-111111111111"

	t.Run("Remember", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusCreated, map[string]any{"written": []string{"likes tea"}, "skipped": []string{}}, rec)
		defer cleanup()
		got, err := c.RememberMemory(ctx, pid, "user1", []string{"likes tea"}, nil, "")
		if err != nil {
			t.Fatalf("RememberMemory: %v", err)
		}
		if rec.method != http.MethodPost || rec.path != "/agent-memory" {
			t.Fatalf("got %s %s", rec.method, rec.path)
		}
		if len(got.Written) != 1 || got.Written[0] != "likes tea" {
			t.Fatalf("written: %+v", got.Written)
		}
		if rec.body["sessionKey"] != "user1" {
			t.Fatalf("sessionKey: %v", rec.body["sessionKey"])
		}
	})

	t.Run("RememberValidation", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{}, rec)
		defer cleanup()
		if _, err := c.RememberMemory(ctx, pid, "user1", nil, nil, ""); err == nil {
			t.Fatal("expected validation error for empty facts+turns")
		}
	})

	t.Run("Recall", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{
			"semantic": []map[string]any{{"content": "likes tea", "score": 0.9}},
		}, rec)
		defer cleanup()
		got, err := c.RecallMemory(ctx, pid, "user1", "tea", 3)
		if err != nil {
			t.Fatalf("RecallMemory: %v", err)
		}
		if rec.method != http.MethodGet {
			t.Fatalf("method: %s", rec.method)
		}
		if !strings.Contains(rec.path, "query=tea") || !strings.Contains(rec.path, "limit=3") {
			t.Fatalf("path: %s", rec.path)
		}
		if len(got.Semantic) != 1 || got.Semantic[0].Content != "likes tea" {
			t.Fatalf("semantic: %+v", got.Semantic)
		}
	})

	t.Run("Forget", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{"forgotten": 2}, rec)
		defer cleanup()
		n, err := c.ForgetMemory(ctx, pid, "user1")
		if err != nil {
			t.Fatalf("ForgetMemory: %v", err)
		}
		if rec.method != http.MethodDelete {
			t.Fatalf("method: %s", rec.method)
		}
		if n != 2 {
			t.Fatalf("forgotten: %d", n)
		}
	})
}

func TestVoiceMethods(t *testing.T) {
	ctx := context.Background()
	pid := "11111111-1111-4111-8111-111111111111"

	t.Run("Transcribe", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{
			"language": "en", "text": "hi",
			"words": []map[string]any{{"word": "hi", "startMs": 0, "endMs": 200}},
		}, rec)
		defer cleanup()
		got, err := c.TranscribeVoice(ctx, pid, "d2F2", "en")
		if err != nil {
			t.Fatalf("TranscribeVoice: %v", err)
		}
		if rec.path != "/voice/transcribe" {
			t.Fatalf("path: %s", rec.path)
		}
		if len(got.Words) != 1 || got.Words[0].Word != "hi" {
			t.Fatalf("words: %+v", got.Words)
		}
		if rec.body["audioBase64"] != "d2F2" {
			t.Fatalf("audioBase64: %v", rec.body["audioBase64"])
		}
	})

	t.Run("Deepfake", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{"probability": 0.97, "model": "x"}, rec)
		defer cleanup()
		got, err := c.ScoreVoiceDeepfake(ctx, pid, "d2F2")
		if err != nil {
			t.Fatalf("ScoreVoiceDeepfake: %v", err)
		}
		if rec.path != "/voice/deepfake-score" {
			t.Fatalf("path: %s", rec.path)
		}
		if got.Probability != 0.97 {
			t.Fatalf("probability: %f", got.Probability)
		}
	})

	t.Run("Validation", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{}, rec)
		defer cleanup()
		if _, err := c.TranscribeVoice(ctx, pid, "", "en"); err == nil {
			t.Fatal("expected validation error for empty audio")
		}
	})
}

func TestLanguageMethods(t *testing.T) {
	ctx := context.Background()
	pid := "11111111-1111-4111-8111-111111111111"

	t.Run("Detect", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{
			"iso6393": "fra", "iso6391": "fr", "name": "French", "confidence": 0.4, "reliable": true,
		}, rec)
		defer cleanup()
		got, err := c.DetectLanguage(ctx, pid, "Bonjour tout le monde")
		if err != nil {
			t.Fatalf("DetectLanguage: %v", err)
		}
		if rec.path != "/language/detect" {
			t.Fatalf("path: %s", rec.path)
		}
		if got.ISO6391 == nil || *got.ISO6391 != "fr" {
			t.Fatalf("iso6391: %v", got.ISO6391)
		}
		if !got.Reliable {
			t.Fatal("expected reliable")
		}
	})

	t.Run("Validation", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{}, rec)
		defer cleanup()
		if _, err := c.DetectLanguage(ctx, pid, ""); err == nil {
			t.Fatal("expected validation error for empty text")
		}
	})
}
