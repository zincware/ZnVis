"""Subprocess entry point for parallel headless rendering workers."""

import json
import os
import pickle
import sys
import traceback

from znvis.visualizer._parallel_render import (
    _initialize_parallel_worker,
    _render_frame_parallel_worker,
)


def _emit(payload: dict) -> None:
    """Emit one JSON message line to stdout for the parent scheduler."""
    print(json.dumps(payload), flush=True)


def _worker_payload(
    status: str, gpu_id: int, cuda_visible_device: str, **extra
) -> dict:
    """Build a scheduler protocol payload for this worker process."""
    payload = {
        "status": status,
        "gpu_id": gpu_id,
        "cuda_visible_device": cuda_visible_device,
        "pid": os.getpid(),
    }
    payload.update(extra)
    return payload


def main() -> int:
    """Run the worker message loop for one GPU-pinned subprocess."""
    state_path, gpu_id_arg = sys.argv[1:3]
    gpu_id = int(gpu_id_arg)
    cuda_visible_device = os.getenv("CUDA_VISIBLE_DEVICES", "")

    with open(state_path, "rb") as file:
        state = pickle.load(file)

    state["gpu_id"] = gpu_id
    state["cuda_visible_device"] = cuda_visible_device
    _initialize_parallel_worker(state)

    if os.getenv("ZNVIS_PARALLEL_DEBUG") == "1":
        print(
            "ZnVis parallel worker "
            f"pid={os.getpid()} gpu_id={gpu_id} "
            f"CUDA_VISIBLE_DEVICES={cuda_visible_device}",
            file=sys.stderr,
            flush=True,
        )

    _emit(_worker_payload("ready", gpu_id, cuda_visible_device))

    for line in sys.stdin:
        message = line.strip()
        if message == "STOP":
            break
        try:
            frame_index = int(message)
            rendered_frame = _render_frame_parallel_worker(frame_index)
            _emit(
                _worker_payload(
                    "ok",
                    gpu_id,
                    cuda_visible_device,
                    frame=rendered_frame,
                )
            )
        except Exception:
            _emit(
                _worker_payload(
                    "error",
                    gpu_id,
                    cuda_visible_device,
                    frame=message,
                    error=traceback.format_exc(),
                )
            )
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
