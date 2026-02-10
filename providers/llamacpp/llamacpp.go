package llamacpp

import (
	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
)

const (
	defaultBaseURL = "http://127.0.0.1:8080/v1"
	providerName   = "llamacpp"
	defaultAPIKey  = "llama-cpp-dummy-key"
)

var (
	_ providers.Provider           = (*Provider)(nil)
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
)

type Provider struct {
	*openai.CompatibleProvider
}

func New(opts ...config.Option) (*Provider, error) {
	defaults := []config.Option{
		config.WithAPIKey(defaultAPIKey),
	}
	opts = append(defaults, opts...)

	base, err := openai.NewCompatible(openai.CompatibleConfig{
		APIKeyEnvVar:   "",
		BaseURLEnvVar:  "",
		Capabilities:   llamacppCapabilities(),
		DefaultAPIKey:  defaultAPIKey,
		DefaultBaseURL: defaultBaseURL,
		Name:           providerName,
		RequireAPIKey:  false,
	}, opts...)
	if err != nil {
		return nil, err
	}

	return &Provider{CompatibleProvider: base}, nil
}

func llamacppCapabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionStreaming: true,
		Embedding:           true,
		ListModels:          true,
	}
}
