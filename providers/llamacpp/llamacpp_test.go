package llamacpp

import (
	"context"
	"testing"
	"time"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testBaseURL                     = "http://127.0.0.1:8080/v1"
	testModel                       = "llama.cpp"
	testLlamacppAvailabilityTimeout = 5 * time.Second
)

func TestNew(t *testing.T) {
	t.Parallel()

	t.Run("creates provider with defaults", func(t *testing.T) {
		p, err := New()
		require.NoError(t, err)
		require.NotNil(t, p)
	})

	t.Run("creates provider with options", func(t *testing.T) {
		p, err := New(anyllm.WithBaseURL("http://custom:8080"))
		require.NoError(t, err)
		require.NotNil(t, p)
	})
}

func TestProviderName(t *testing.T) {
	t.Parallel()
	p, err := New()
	require.NoError(t, err)
	assert.Equal(t, providerName, p.Name())
}

func TestCapabilities(t *testing.T) {
	t.Parallel()
	p, err := New()
	require.NoError(t, err)
	caps := p.Capabilities()
	assert.True(t, caps.Completion)
	assert.True(t, caps.CompletionStreaming)
	assert.True(t, caps.Embedding)
	assert.True(t, caps.ListModels)
}

func TestIntegration_Llamacpp(t *testing.T) {
	t.Parallel()
	skipIfLlamacppUnavailable(t)

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
		require.NotEmpty(t, resp.Choices[0].Message.Content)
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

func skipIfLlamacppUnavailable(t *testing.T) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), testLlamacppAvailabilityTimeout)
	defer cancel()

	p, err := New()
	if err != nil {
		t.Skip("llamacpp not available: failed to create provider")
	}

	if _, err = p.ListModels(ctx); err != nil {
		t.Skip("llamacpp not available: server not responding at " + testBaseURL)
	}
}
