# Local development

Run commands from `platform/`. The existing `uv` executable is the only prerequisite;
setup does not install or upgrade it.

```sh
./dev setup
./dev doctor
./dev test
./dev lint
./dev generate --prompt "In machine learning, explain a linear probe in two sentences." \
  --device mps --max-new-tokens 96 --output artifacts/mps-smoke.json
```

## Isolation

`./dev setup` downloads a private CPython 3.12.11 into `.python/` and installs the
exact `uv.lock` dependencies into `.venv/`. Package downloads, Hugging Face model
weights, and Torch caches live in `.cache/`. None of these directories are tracked.
The launcher ignores an activated environment and Python import-path overrides,
disables user-site packages, and verifies its interpreter before execution.

No system/Homebrew Python packages, parent-project dependencies, shell startup
files, global Python symlinks, or Poetry configuration are changed. Do not use
bare `pip`, `poetry install`, or `uv sync` for this project; use `./dev setup`.
`./dev lock` is the explicit dependency-resolution command; review and commit the
resulting lockfile when changing requirements. No environment activation is needed.

## Runtime and measurements

PyTorch is pinned to 2.7.1 for this Mac's macOS 14.3: the initially resolved 2.15.0
aborted inside a Metal linear operation. Both a tiny-model MPS test and a full
Qwen generation validate the pinned version. CUDA support is implemented but has
not been exercised on this Mac.

The default model is `Qwen/Qwen3-4B-Instruct-2507`. The first run downloads about
8 GB of weights. MPS/CUDA use FP16; CPU uses FP32 and needs roughly 16 GB for the
default model's weights alone. Start with batch size one and short prompts.
`--device auto` selects CUDA, then MPS, then CPU; use an explicit device to prevent
unexpected CPU execution. Remote NVIDIA runs use `--device cuda` with the same CLI.

Each JSON report records the resolved model commit, package versions, frozen-weight
status, download/load time, warmup time, and generation throughput including prompt
prefill. Pass `--revision <commit>` to reproduce a specific checkpoint. Generation
uses greedy decoding and has a default 32-token output limit. Prompts exceeding
the input limit are rejected rather than silently truncated.

Memory sampling spans loading, warmup, and generation at 20 ms intervals. MPS
reports sampled peaks for tensor allocations and driver allocations (including
allocator caches), alongside sampled process RSS. These are lower bounds and may
miss short spikes. Apple unified-memory figures overlap: do not add them together.
CUDA additionally reports its exact peak allocated tensor memory. Tests use a
tiny locally constructed Qwen model and do not download pretrained weights.

Activation capture, probe training, and serving are subsequent milestones.

## Initial local measurement

On the M3 with 18 GB unified memory and macOS 14.3, the command above produced
79 tokens in 9.11 seconds (~8.7 tokens/second including prefill), after a 7.46-second
cached load and a 1.18-second warmup. Sampled MPS tensor memory peaked at 7.52 GiB;
driver memory peaked at 8.62 GiB. System memory pressure remained normal, although
some swap was used. This is a short-request smoke test, not a long-context capacity
benchmark or a quality evaluation. Detailed local results are in the ignored
`artifacts/mps-smoke.json` file.
