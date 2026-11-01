"""Memory measurements with explicit sampling semantics on Apple GPUs."""

from threading import Event, Thread

import psutil
import torch


class MemorySampler:
    """Sample whole-process RSS and MPS allocations; CUDA also has an exact allocator peak.

    Samples are lower bounds on peaks, not promises that brief spikes were observed.
    On Apple silicon RSS and GPU memory overlap; they must not be added together.
    """

    def __init__(self, device: torch.device, interval: float = 0.02):
        self.device = device
        self.interval = interval
        self.peaks: dict[str, int] = {}
        self.process = psutil.Process()
        self.stop_event = Event()
        self.thread: Thread | None = None

    def _sample(self) -> None:
        values = {"process_rss_sampled_peak_bytes": self.process.memory_info().rss}
        if self.device.type == "mps":
            values.update(
                mps_tensor_sampled_peak_bytes=torch.mps.current_allocated_memory(),
                mps_driver_sampled_peak_bytes=torch.mps.driver_allocated_memory(),
            )
        for key, value in values.items():
            self.peaks[key] = max(self.peaks.get(key, 0), value)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            self._sample()

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sample()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self._sample()
        if self.device.type == "cuda":
            self.peaks["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)

    def report(self) -> dict:
        return {"sample_interval_seconds": self.interval, **self.peaks}
