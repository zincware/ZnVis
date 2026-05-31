"""Parallel rendering utilities for the headless visualizer."""

import json
import logging
import multiprocessing as mp
import os
import pickle
import selectors
import subprocess
import sys
import tempfile
import warnings
from copy import copy
from dataclasses import dataclass

logger = logging.getLogger(__name__)
_PARALLEL_RENDER_STATE = {}


@dataclass(frozen=True)
class _WorkerSpec:
    """CUDA device assignment for one worker process."""

    gpu_id: int
    cuda_visible_device: str


def _visible_cuda_devices(available_gpu_devices: int) -> list[str]:
    """Return CUDA device identifiers visible to the parent process."""
    visible_gpus = int(available_gpu_devices)
    if visible_gpus <= 0:
        return []

    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        devices = [d.strip() for d in cuda_visible_devices.split(",")]
        devices = [d for d in devices if d and d != "-1"]
        return devices[:visible_gpus]

    return [str(device_id) for device_id in range(visible_gpus)]


def _build_worker_specs(
    parallel_render_workers: int, available_gpu_devices: int
) -> list[_WorkerSpec]:
    """Assign CUDA workers to visible GPUs.

    EXPERIMENTAL FEATURE
    -----
    When ``parallel_render_workers`` is larger than the number of visible
    GPUs, workers are assigned round-robin across devices.
    """
    total_workers = int(parallel_render_workers)
    devices = _visible_cuda_devices(available_gpu_devices)
    if not devices:
        raise RuntimeError("Parallel rendering requires at least one visible CUDA GPU.")

    if total_workers > len(devices):
        warnings.warn(
            "Using more parallel render workers than visible GPUs is experimental. "
            "Workers will be distributed round-robin over the available GPUs.",
            UserWarning,
            stacklevel=2,
        )

    return [
        _WorkerSpec(
            gpu_id=worker_id % len(devices),
            cuda_visible_device=devices[worker_id % len(devices)],
        )
        for worker_id in range(total_workers)
    ]


def make_spawn_safe_render_items(items):
    """Create lightweight copies for worker transfer by dropping mesh lists."""
    if items is None:
        return None
    safe_items = []
    for item in items:
        item_copy = copy(item)
        item_copy.mesh_list = None
        safe_items.append(item_copy)
    return safe_items


def _worker_processes(workers: list[dict]) -> list[subprocess.Popen]:
    """Return process handles from worker dictionaries."""
    return [worker["process"] for worker in workers]


def _wait_or_kill_processes(processes, timeout: float = 1.0) -> None:
    """Wait for processes to exit and kill stragglers after timeout."""
    for proc in processes:
        if proc is None:
            continue
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()


def _force_stop_processes(processes) -> None:
    """Best-effort hard stop for worker subprocesses."""
    for proc in processes:
        if proc is not None and proc.poll() is None:
            proc.terminate()
    _wait_or_kill_processes(processes, timeout=1.0)


def _render_frame_parallel_worker(frame_index: int) -> int:
    """Render one frame inside a worker process."""
    state = _PARALLEL_RENDER_STATE
    renderer = state.get("renderer")
    if renderer is None:
        from znvis.rendering import Mitsuba

        renderer = Mitsuba()
        state["renderer"] = renderer

    from znvis.visualizer.base_visualizer import build_mesh_dict_for_frame

    mesh_dict = build_mesh_dict_for_frame(
        particles=state["particles"],
        vector_field=state["vector_field"],
        frame_index=frame_index,
    )
    view_matrix = (
        state["camera"].get_view_matrix(frame_index)
        if state["camera"] is not None
        else state["view_matrix"]
    )

    renderer.render_mesh_objects(
        mesh_dict,
        view_matrix,
        save_dir=state["frame_folder"],
        save_name=f"frame_{frame_index:0>6}.png",
        resolution=state["renderer_resolution"],
        samples_per_pixel=state["renderer_spp"],
    )
    return frame_index


def _initialize_parallel_worker(state: dict):
    """Initialize per-process rendering state for parallel worker execution."""
    global _PARALLEL_RENDER_STATE
    _PARALLEL_RENDER_STATE = dict(state)
    _PARALLEL_RENDER_STATE["renderer"] = None


def _start_worker_process(spec: _WorkerSpec, state_path: str):
    """Start one isolated CUDA worker subprocess pinned to one visible GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.cuda_visible_device
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "znvis.visualizer.parallel.parallel_render_worker",
            state_path,
            str(spec.gpu_id),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )


def _send_frame(worker: dict, frame_index: int) -> None:
    """Send one global frame index to a worker process."""
    worker["process"].stdin.write(f"{frame_index}\n")
    worker["process"].stdin.flush()


def _stop_worker(worker: dict) -> None:
    """Request graceful worker shutdown via the stdin control channel."""
    process = worker["process"]
    if process.poll() is not None or process.stdin.closed:
        return
    process.stdin.write("STOP\n")
    process.stdin.flush()


def _wait_until_ready(selector, workers: list[dict]) -> None:
    """Wait until all workers emit a ready status."""
    ready_workers = 0
    while ready_workers < len(workers):
        events = selector.select(timeout=1)
        if not events:
            if any(worker["process"].poll() is not None for worker in workers):
                raise RuntimeError(
                    "A parallel render worker exited before becoming ready."
                )
            continue

        for key, _mask in events:
            line = key.fileobj.readline()
            if line == "":
                raise RuntimeError(
                    "A parallel render worker closed stdout before becoming ready."
                )
            payload = json.loads(line)
            if payload["status"] != "ready":
                raise RuntimeError(
                    f"Unexpected parallel worker startup message: {payload}"
                )
            ready_workers += 1


def render_frames_parallel(
    render_config: dict, progress_factory, frame_indices=None
) -> None:
    """
    Render selected global frames with isolated CUDA worker subprocesses.

    Parameters
    ----------
    render_config : dict
        Dictionary providing explicit config parameters and states.
    progress_factory : callable
        Context-manager factory that returns a progress object with ``add_task`` and
        ``update`` methods (e.g., ``rich.progress.Progress``).
    frame_indices : sequence[int] | None, optional
        Explicit global frame indices to render. If ``None``, all frames from
        ``range(number_of_steps)`` are scheduled.
    """
    number_of_steps = render_config["number_of_steps"]
    parallel_render_workers = render_config["parallel_render_workers"]
    available_gpu_devices = render_config["available_gpu_devices"]
    base_worker_state = render_config["worker_state"]

    selected_frame_indices = (
        list(range(number_of_steps)) if frame_indices is None else list(frame_indices)
    )
    if not selected_frame_indices:
        return

    try:
        worker_specs = _build_worker_specs(
            parallel_render_workers, available_gpu_devices
        )
    except RuntimeError as e:
        raise RuntimeError(f"Parallel rendering requirements failed: {e}")

    if parallel_render_workers == 1:
        raise ValueError(
            "Parallel rendering invoked with only 1 worker worker configuration."
        )

    if mp.current_process().name != "MainProcess":
        raise RuntimeError(
            "Parallel rendering must be started from the main process. "
            "Ensure your entry script only calls render code under "
            "if __name__ == '__main__'."
        )

    workers: list[dict] = []
    selector = selectors.DefaultSelector()
    interrupted = False
    try:
        with tempfile.TemporaryDirectory(prefix="znvis-render-state-") as temp_dir:
            state_path = os.path.join(temp_dir, "state.pkl")
            with open(state_path, "wb") as file:
                pickle.dump(base_worker_state, file)

            for spec in worker_specs:
                process = _start_worker_process(spec, state_path)
                worker = {"process": process}
                workers.append(worker)
                selector.register(process.stdout, selectors.EVENT_READ, worker)

            _wait_until_ready(selector, workers)

            next_frame_position = 0
            completed_frames = 0
            for worker in workers:
                if next_frame_position >= len(selected_frame_indices):
                    break
                _send_frame(worker, selected_frame_indices[next_frame_position])
                next_frame_position += 1

            with progress_factory() as progress:
                task = progress.add_task(
                    "Saving scenes...", total=len(selected_frame_indices)
                )
                while completed_frames < len(selected_frame_indices):
                    events = selector.select(timeout=1)
                    if not events:
                        if all(
                            worker["process"].poll() is not None for worker in workers
                        ):
                            raise RuntimeError(
                                "All parallel render workers exited before rendering "
                                "completed."
                            )
                        continue

                    for key, _mask in events:
                        worker = key.data
                        line = key.fileobj.readline()
                        if line == "":
                            selector.unregister(key.fileobj)
                            continue

                        payload = json.loads(line)
                        if payload["status"] == "error":
                            raise RuntimeError(
                                "Parallel render worker failed "
                                f"pid={payload['pid']} "
                                f"gpu_id={payload['gpu_id']} "
                                f"frame={payload['frame']}:\n{payload['error']}"
                            )

                        completed_frames += 1
                        progress.update(task, advance=1)

                        if next_frame_position < len(selected_frame_indices):
                            _send_frame(
                                worker, selected_frame_indices[next_frame_position]
                            )
                            next_frame_position += 1

            for worker in workers:
                _stop_worker(worker)
    except KeyboardInterrupt:
        interrupted = True
        logger.warning(
            "Parallel rendering interrupted by user; cancelling pending tasks."
        )
        _force_stop_processes(_worker_processes(workers))
        raise
    except Exception:
        logger.exception(
            "Parallel rendering failed, likely due to worker initialization "
            "or frame-state serialization."
        )
        _force_stop_processes(_worker_processes(workers))
        raise
    finally:
        if not interrupted:
            live_processes = [
                process
                for process in _worker_processes(workers)
                if process.poll() is None
            ]
            _wait_or_kill_processes(live_processes, timeout=1.0)
