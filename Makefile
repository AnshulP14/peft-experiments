PY := ../.venv/bin/python
MLFLOW_URI := http://127.0.0.1:5000

.PHONY: install serve serve-stop qlora qlora-smoke tune tune-smoke overnight

install:
	../.venv/bin/pip install -r requirements.txt

serve:            ## start MLflow tracking server + UI on :5000
	mkdir -p output
	@nohup $(PY) -m mlflow server --backend-store-uri "sqlite://$(CURDIR)/output/mlruns.db" \
		--default-artifact-root "$(CURDIR)/output/mlflow-artifacts" \
		--host 127.0.0.1 --port 5000 > output/mlflow-server.log 2>&1 &
	@sleep 6 && echo "MLflow UI: $(MLFLOW_URI)  (log: output/mlflow-server.log)"

serve-stop:
	pkill -f "mlflow server" || true

qlora:            ## full QLoRA run (env/CLI config; see .env.example)
	$(PY) methods/qlora.py

qlora-smoke:      ## ~1 min smoke test (64 rows, 1 epoch)
	PEFT_NUM_TRAIN=64 PEFT_EPOCHS=1 $(PY) methods/qlora.py

tune:             ## Optuna study: 50 trials / 12h (resumable; --study to name it)
	$(PY) tune.py --n_trials 50 --timeout 12h

tune-smoke:       ## tiny study (3 trials, small data) to validate the pipeline
	PEFT_TUNE_NUM_TRAIN=64 PEFT_TUNE_EPOCHS=1 PEFT_TUNE_MAX_LENGTH=256 \
	$(PY) tune.py --n_trials 3 --timeout 15m --repeats 1 --n_startup_trials 2

overnight:        ## detached overnight run (see run_overnight.sh)
	./run_overnight.sh
