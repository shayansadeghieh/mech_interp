import torch

from activation_monitor.memory import MemorySampler


def test_sampler_stops_even_when_inference_fails():
    sampler = MemorySampler(torch.device("cpu"))
    try:
        with sampler:
            raise RuntimeError("failed request")
    except RuntimeError:
        pass
    assert not sampler.thread.is_alive()
    assert sampler.report()["process_rss_sampled_peak_bytes"] > 0
    assert "cuda_peak_allocated_bytes" not in sampler.report()
