MOD ?= 07-agents-mcp-evals
MODULES := 01-delivery-sla-breach 02-classic-nlp 03-transformer-onnx 04-multimodal-ai 05-llm-gateway 06-production-rag 07-agents-mcp-evals

ORT_DIR := $(CURDIR)/.ort
ORT_SO  := $(ORT_DIR)/onnxruntime.so
PID_DIR := $(ORT_DIR)/pids

.PHONY: infra-up infra-down infra-logs ort-setup train notebook gen gen-all generate generate-all test test-all serve serve-all stop-all demo-traffic

infra-up:
	docker compose up -d

infra-down:
	docker compose down

infra-logs:
	docker compose logs -f

ort-setup:
	@mkdir -p "$(ORT_DIR)"
	@if [ ! -e "$(ORT_SO)" ]; then \
		rm -f "$(ORT_SO)" "$(ORT_DIR)/libonnxruntime.so"; \
		LIB=$$(find $(CURDIR)/0*/ml/.venv /usr/local /usr/lib* /opt $(HOME)/.local -name 'libonnxruntime.so*' \( -type f -o -type l \) 2>/dev/null | head -n 1); \
		if [ -n "$$LIB" ]; then \
			REAL_LIB=$$(readlink -f "$$LIB"); \
			ln -sf "$$REAL_LIB" "$(ORT_SO)"; \
			ln -sf "$$REAL_LIB" "$(ORT_DIR)/libonnxruntime.so"; \
			echo "Linked $(ORT_SO) -> $$REAL_LIB"; \
		else \
			echo "ERROR: libonnxruntime.so* not found. Run: (cd 01-delivery-sla-breach/ml && uv sync)"; \
			exit 1; \
		fi; \
	fi

train:
	cd $(MOD)/ml && uv run python train.py

notebook:
	cd $(MOD)/ml && uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

gen: generate

gen-all: generate-all

generate:
	cd $(MOD)/go && GOWORK=off go generate ./...

generate-all:
	@for m in $(MODULES); do \
		echo "=== Generating contract for $$m ==="; \
		(cd $$m/go && GOWORK=off go generate ./...) || exit 1; \
	done

test: ort-setup
	@ORT_REAL_DIR=$$(dirname "$$(readlink -f "$(ORT_SO)")"); \
	cd $(MOD)/go && GOWORK=off ONNXRUNTIME_LIB="$(ORT_SO)" LD_LIBRARY_PATH="$(ORT_DIR):$$ORT_REAL_DIR$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH}" go test -count=1 -v ./...

test-all: ort-setup
	@ORT_REAL_DIR=$$(dirname "$$(readlink -f "$(ORT_SO)")"); \
	for m in $(MODULES); do \
		echo "=== Testing $$m ==="; \
		(cd $$m/go && GOWORK=off ONNXRUNTIME_LIB="$(ORT_SO)" LD_LIBRARY_PATH="$(ORT_DIR):$$ORT_REAL_DIR$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH}" go test -count=1 ./...) || exit 1; \
	done

serve: ort-setup
	@ORT_REAL_DIR=$$(dirname "$$(readlink -f "$(ORT_SO)")"); \
	cd $(MOD)/go && GOWORK=off ONNXRUNTIME_LIB="$(ORT_SO)" LD_LIBRARY_PATH="$(ORT_DIR):$$ORT_REAL_DIR$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH}" go run ./cmd/server

serve-all: ort-setup stop-all
	@mkdir -p "$(PID_DIR)"
	@ORT_REAL_DIR=$$(dirname "$$(readlink -f "$(ORT_SO)")"); \
	for m in $(MODULES); do \
		echo "Building & starting $$m ..."; \
		(cd $$m/go && GOWORK=off go build -o "$(PID_DIR)/$$m.bin" ./cmd/server) || exit 1; \
		(cd $$m/go && \
			ONNXRUNTIME_LIB="$(ORT_SO)" \
			LD_LIBRARY_PATH="$(ORT_DIR):$$ORT_REAL_DIR$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH}" \
			DATABASE_URL="postgres://apprentice:apprentice_dev@localhost:5432/rag?sslmode=disable" \
			RUSTFS_ENDPOINT="http://localhost:9000" \
			RAG_SERVICE_URL="http://localhost:8086" \
			"$(PID_DIR)/$$m.bin" > "$(PID_DIR)/$$m.log" 2>&1 & echo $$! > "$(PID_DIR)/$$m.pid"); \
	done
	@echo "All 7 Go AI services started (:8081 -> :8087). Logs in .ort/pids/*.log"

stop-all:
	@if [ -d "$(PID_DIR)" ]; then \
		for pidfile in "$(PID_DIR)"/*.pid; do \
			[ -f "$$pidfile" ] && kill $$(cat "$$pidfile") 2>/dev/null || true; \
		done; \
		rm -f "$(PID_DIR)"/*.pid; \
	fi

demo-traffic:
	python3 infra/replay.py
