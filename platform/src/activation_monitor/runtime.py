"""Frozen Transformers runtime shared by local MPS and remote CUDA execution."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            "MPS is unavailable. Choose --device cpu or use an Apple GPU with MPS support."
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Choose --device mps or cpu on this machine.")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str = DEFAULT_MODEL
    revision: str = "main"
    cache_dir: Path = Path(".cache/huggingface/hub")


class FrozenModel:
    def __init__(self, model, tokenizer, device: torch.device, revision: str):
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.device = device
        self.revision = revision

    @classmethod
    def load(cls, spec: ModelSpec, device: torch.device):
        kwargs = {"cache_dir": str(spec.cache_dir), "trust_remote_code": False}
        config = AutoConfig.from_pretrained(spec.model_id, revision=spec.revision, **kwargs)
        # Resolve a floating revision once, then use the same commit for all artifacts.
        revision = config._commit_hash or spec.revision
        tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=revision, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            revision=revision,
            config=config,
            dtype=torch.float32 if device.type == "cpu" else torch.float16,
            attn_implementation="sdpa",
            use_safetensors=True,
            **kwargs,
        ).to(device)
        synchronize(device)
        return cls(model, tokenizer, device, revision)

    def encode(self, prompt: str, max_input_tokens: int) -> dict[str, torch.Tensor]:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False
        )
        length = encoded["input_ids"].shape[-1]
        if length > max_input_tokens:
            raise ValueError(
                f"Prompt has {length} tokens; limit is {max_input_tokens}. "
                "Shorten the prompt or increase --max-input-tokens. Nothing was truncated."
            )
        return {key: value.to(self.device) for key, value in encoded.items()}

    @torch.inference_mode()
    def generate(self, encoded: dict[str, torch.Tensor], max_new_tokens: int):
        return self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
            use_cache=True,
        )

    def benchmark(
        self, prompt: str, max_input_tokens: int, max_new_tokens: int, warmup_tokens: int
    ):
        encoded = self.encode(prompt, max_input_tokens)
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        prompt_tokens = encoded["input_ids"].shape[-1]
        if context_limit and prompt_tokens + max(max_new_tokens, warmup_tokens) > context_limit:
            raise ValueError(
                f"Requested prompt and output exceed model context limit {context_limit}."
            )
        warmup_started = perf_counter()
        if warmup_tokens:
            self.generate(encoded, warmup_tokens)
            synchronize(self.device)
        warmup_seconds = perf_counter() - warmup_started
        synchronize(self.device)
        started = perf_counter()
        output = self.generate(encoded, max_new_tokens)
        synchronize(self.device)
        seconds = perf_counter() - started
        generated = output[0, prompt_tokens:]
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "input_tokens": prompt_tokens,
            "generated_tokens": generated.numel(),
            "generation_seconds": seconds,
            "tokens_per_second_including_prefill": generated.numel() / seconds,
            "warmup_seconds": warmup_seconds,
            "warmup_max_new_tokens": warmup_tokens,
        }
