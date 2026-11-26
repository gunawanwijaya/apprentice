MOD ?= 01-delivery-sla-breach
ORT_LIB := $(shell find $(MOD)/ml/.venv -name "libonnxruntime.so*" 2>/dev/null | head -n 1 | xargs -r realpath)

.PHONY: notebook train gen test

notebook:
	cd $(MOD)/ml && uv run jupyter lab

train:
	cd $(MOD)/ml && uv run python train.py

gen:
	cd $(MOD)/go && GOWORK=off go generate ./...

test:
	cd $(MOD)/go && ONNXRUNTIME_LIB="$(if $(ORT_LIB),$(realpath $(ORT_LIB)),)" GOWORK=off go test -count=1 -v ./...

serve:
	cd $(MOD)/go && ONNXRUNTIME_LIB="$(if $(ORT_LIB),$(realpath $(ORT_LIB)),)" GOWORK=off go run ./cmd/server  | jq -R 'fromjson? // .'
