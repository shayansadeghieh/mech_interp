"""CLI for environment inspection and a measured single-request generation."""

import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch

from activation_monitor.memory import MemorySampler
from activation_monitor.runtime import DEFAULT_MODEL, FrozenModel, ModelSpec, resolve_device


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "environment": sys.prefix,
        "base_python": sys.base_prefix,
        "torch": version("torch"),
        "transformers": version("transformers"),
        "torch_path": torch.__file__,
        "huggingface_cache": os.environ.get("HF_HUB_CACHE", str(ModelSpec().cache_dir)),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="activation-monitor")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Show interpreter, dependency paths, and available devices")
    generate = commands.add_parser("generate", help="Run and measure one frozen-model request")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--model", default=DEFAULT_MODEL)
    generate.add_argument("--revision", default="main")
    generate.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    generate.add_argument("--max-input-tokens", type=positive_int, default=512)
    generate.add_argument("--max-new-tokens", type=positive_int, default=32)
    generate.add_argument("--warmup-tokens", type=nonnegative_int, default=8)
    generate.add_argument("--output", type=Path, help="Also write the JSON report to this file")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(environment(), indent=2))
        return 0
    try:
        if not args.prompt.strip():
            raise ValueError("Prompt must not be empty.")
        device = resolve_device(args.device)
        if device.type == "cpu":
            print(
                "CPU mode uses FP32 weights; the default 4B model needs ~16 GB for weights alone.",
                file=sys.stderr,
            )
        spec = ModelSpec(args.model, args.revision)
        print(
            f"Loading {spec.model_id} on {device} (first run downloads model weights)…",
            file=sys.stderr,
            flush=True,
        )
        with MemorySampler(device) as memory:
            started = perf_counter()
            runtime = FrozenModel.load(spec, device)
            load_seconds = perf_counter() - started
            print("Model loaded; warming up and measuring generation…", file=sys.stderr, flush=True)
            result = runtime.benchmark(
                args.prompt, args.max_input_tokens, args.max_new_tokens, args.warmup_tokens
            )
        report = {
            "model": spec.model_id,
            "resolved_revision": runtime.revision,
            "device": str(device),
            "dtype": str(runtime.model.dtype),
            "parameters_frozen": all(not p.requires_grad for p in runtime.model.parameters()),
            "download_and_load_seconds": load_seconds,
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
            "prompt": args.prompt,
            **result,
            "memory": memory.report(),
            "environment": environment(),
        }
        serialized = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n")
        print(serialized)
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
