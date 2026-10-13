PYTHON ?= python
COMPOSE ?= docker compose

.PHONY: eval-openai eval-local eval-local-preflight use-openai use-local

eval-openai:
	$(PYTHON) scripts/eval_model_profile.py openai-gpt-4o-mini

eval-local:
	$(PYTHON) scripts/eval_model_profile.py ollama-qwen3-8b

eval-local-preflight:
	$(PYTHON) scripts/eval_model_profile.py ollama-qwen3-8b --skip-faithfulness-judge

use-openai:
	$(COMPOSE) --env-file .env --env-file config/llm/openai-gpt-4o-mini.env up -d --build ml-api

use-local:
	$(COMPOSE) --env-file .env --env-file config/llm/ollama-qwen3-8b.env up -d --build ml-api
