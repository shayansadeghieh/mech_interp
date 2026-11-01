# Activation Monitoring Platform

**Design proposal · MVP**

Turn LLM activations into validated, versioned monitors. Users harvest activations, train probes for specific behaviors, and deploy those probes alongside inference to publish metrics to Prometheus and Grafana. Probe scores are predictive signals; they do not by themselves establish a model’s intent or explain its mechanisms.

## User workflow

1. **Capture:** Run labeled tasks through a frozen model and save selected layer activations, token positions, inputs, outputs, and observed outcomes.
2. **Train:** Fit linear probes on captured activations. Compare layers and token aggregation strategies against text-based baselines.
3. **Validate:** Evaluate on held-out task and attack families, including benign lookalikes. Measure recall at a fixed false-positive rate, calibration, detection timing, and runtime overhead.
4. **Deploy:** Run the same monitor during live inference in shadow mode. Export aggregate metrics and retain sampled request-level scores for investigation.
5. **Improve:** Review flagged requests and an unbiased traffic sample; feed labeled examples into the next offline evaluation cycle.

## Architecture

```text
Python inference worker → sampled activations → dataset store
                                                 ↓
                                        train + evaluate
                                                 ↓
Live inference → in-process probe ← versioned monitor registry
                        ├→ aggregate metrics → Prometheus → Grafana
                        └→ request-level score events → investigation
```

Score activations on the model’s device and export small results. Persist raw activations selectively; continuous full-layer, full-token capture is not the default. Keep request IDs and token-level records out of Prometheus labels.

A monitor artifact binds probe weights, threshold, and evaluation results to the exact model revision, adapter, precision, tokenizer/chat template, hook location, token selection, preprocessing, and aggregation. Offline evaluation and online scoring share this specification. Model or capture changes require revalidation.

## Initial implementation

- **Model/runtime:** Frozen `Qwen/Qwen3-4B-Instruct-2507` with Transformers and PyTorch. Start on the M3 Mac using MPS, batch size one, short sequences, and selective capture; measure memory before choosing remote CUDA compute.
- **Learning/storage:** Scikit-learn logistic regression; activation shards and metadata on local disk behind a storage interface. Train probes separately from the LLM.
- **Interface:** Python SDK and CLI for capture, training, evaluation, and deployment. Add an API/MCP interface for assistant-driven experiments after the workflow is stable.
- **First target:** Detect task violations caused by prompt injection in retrieved content, using sandboxed tasks with observable success/failure labels. Evaluate scores available before the monitored action.

## Scope and success

Ship one model, one detector type, and one complete offline-to-online workflow. Success means reproducible artifacts, matching offline/online scores, and measurable detection or cost benefits over a text baseline within an explicit latency budget. Report failures and uncertainty alongside scores.

Defer full-model training, automated blocking, nonlinear detectors, checkpoint comparison, and distributed orchestration. Keep interfaces extensible to these; introduce Ray, Kubernetes, or a Go control plane when workload or deployment needs justify them.
