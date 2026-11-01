"""Exercise the real Transformers path with a tiny, local Qwen model; no downloads."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from activation_monitor.runtime import FrozenModel, ModelSpec, resolve_device


def tiny_model(tmp_path):
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "<pad>": 1, "<eos>": 2, "hello": 3}))
    tokenizer.pre_tokenizer = Whitespace()
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        chat_template="{% for message in messages %}{{ message['content'] }} {% endfor %}",
    )
    wrapped.save_pretrained(tmp_path)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=4,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            pad_token_id=1,
            eos_token_id=2,
        )
    )
    model.save_pretrained(tmp_path)
    return FrozenModel.load(ModelSpec(str(tmp_path)), torch.device("cpu"))


def test_real_model_is_frozen_and_generates(tmp_path):
    runtime = tiny_model(tmp_path)
    assert not runtime.model.training
    assert all(not p.requires_grad for p in runtime.model.parameters())
    encoded = runtime.encode("hello", max_input_tokens=8)
    output = runtime.generate(encoded, max_new_tokens=3)
    assert 2 <= output.shape[-1] <= 4
    assert all(p.grad is None for p in runtime.model.parameters())
    result = runtime.benchmark("hello", 8, 3, 1)
    assert result["input_tokens"] == 1
    assert 1 <= result["generated_tokens"] <= 3
    assert result["tokens_per_second_including_prefill"] > 0


def test_oversize_prompt_is_rejected_without_truncation(tmp_path):
    import pytest

    runtime = tiny_model(tmp_path)
    with pytest.raises(ValueError, match="Nothing was truncated"):
        runtime.encode("hello hello hello", max_input_tokens=2)
    with pytest.raises(ValueError, match="context limit"):
        runtime.benchmark("hello", 8, 32, 0)


def test_unavailable_device_fails_explicitly(monkeypatch):
    import pytest

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="MPS is unavailable"):
        resolve_device("mps")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires an Apple GPU")
def test_tiny_model_generates_on_mps(tmp_path):
    runtime = tiny_model(tmp_path)
    runtime.model.to(device="mps", dtype=torch.float16)
    runtime.device = torch.device("mps")
    result = runtime.benchmark("hello", 8, 3, 1)
    assert result["generated_tokens"] > 0
    torch.mps.synchronize()
