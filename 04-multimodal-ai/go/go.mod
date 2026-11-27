module github.com/gunawanwijaya/apprentice/04-multimodal-ai/go

go 1.27.4

require (
	github.com/yalue/onnxruntime_go v1.38.0
	golang.org/x/sync v0.25.0
)

replace github.com/gunawanwijaya/apprentice/infra/telemetry => ../../infra/telemetry
