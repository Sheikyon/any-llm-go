package llamacpp

import (
	"context"
	"net/http"
	"testing"
	"time"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/stretchr/testify/require"
)

const testBaseURL = "http://127.0.0.1:8080/v1"
const testModel = "llama.cpp"

func TestNew(t *testing.T) {
	t.Run("creates provider with defaults", func(t *testing.T) {
		p, err := New()
		require.NoError(t, err)
		require.NotNil(t, p)
	})
}

func serverRunning(t *testing.T) bool {
	t.Helper()
	client := &http.Client{Timeout: 1 * time.Second}
	resp, err := client.Get(testBaseURL + "/models")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func TestIntegration_Llamacpp(t *testing.T) {
	if !serverRunning(t) {
		t.Skip("No llama.cpp server running")
	}

	p, err := New(anyllm.WithTimeout(30 * time.Second))
	require.NoError(t, err)
	ctx := context.Background()

	t.Run("ListModels", func(t *testing.T) {
		models, err := p.ListModels(ctx)
		require.NoError(t, err)
		require.NotEmpty(t, models.Data)
	})

	t.Run("Completion", func(t *testing.T) {
		resp, err := p.Completion(ctx, anyllm.CompletionParams{
			Model:    testModel,
			Messages: testutil.MessagesWithSystem(),
		})
		require.NoError(t, err)
		content, ok := resp.Choices[0].Message.Content.(string)
		require.True(t, ok)
		require.NotEmpty(t, content)
	})

	t.Run("Embedding", func(t *testing.T) {
		resp, err := p.Embedding(ctx, anyllm.EmbeddingParams{
			Model: testModel,
			Input: []string{"test"},
		})
		require.NoError(t, err)
		require.NotEmpty(t, resp.Data)
	})
}
