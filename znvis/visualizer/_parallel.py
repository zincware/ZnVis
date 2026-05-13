"""
Parallel rendering utilities for the headless visualizer.
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy

from znvis.rendering import Mitsuba
from znvis.visualizer.base_visualizer import build_mesh_dict_for_frame

logger = logging.getLogger(__name__)

_PARALLEL_RENDER_STATE = {}


def _make_spawn_safe_render_items(items):
    """
    Create lightweight copies for worker transfer by dropping prebuilt mesh lists.
    """
    safe_items = []
    for item in items:
        item_copy = copy(item)
        item_copy.mesh_list = None
        safe_items.append(item_copy)
    return safe_items


def _force_stop_executor_processes(executor: ProcessPoolExecutor) -> None:
    """
    Best-effort hard stop for worker processes to avoid lingering GPU workers.
    """
    processes = getattr(executor, "_processes", None)
    if not processes:
        return
    for proc in list(processes.values()):
        if proc is None:
            continue
        if proc.is_alive():
            proc.terminate()
    for proc in list(processes.values()):
        if proc is None:
            continue
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()


def _force_stop_active_children() -> None:
    """
    Best-effort cleanup for any remaining multiprocessing child processes.
    """
    for proc in mp.active_children():
        try:
            proc.terminate()
        except Exception:
            logger.exception("Failed to terminate child process pid=%s", proc.pid)
    for proc in mp.active_children():
        try:
            proc.join(timeout=1)
            if proc.is_alive():
                proc.kill()
        except Exception:
            logger.exception("Failed to kill child process pid=%s", proc.pid)


def _render_frame_parallel_worker(frame_index: int) -> int:
    """
    Render one frame inside a worker process.
    """
    state = _PARALLEL_RENDER_STATE
    renderer = state.get("renderer")
    if renderer is None:
        renderer = Mitsuba()
        state["renderer"] = renderer

    mesh_dict = build_mesh_dict_for_frame(
        particles=state["particles"],
        vector_field=state["vector_field"],
        frame_index=frame_index,
    )
    if state["camera"] is not None:
        view_matrix = state["camera"].get_view_matrix(frame_index)
    else:
        view_matrix = state["view_matrix"]

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
    """
    Initialize per-process rendering state for parallel worker execution.
    """
    global _PARALLEL_RENDER_STATE
    _PARALLEL_RENDER_STATE = dict(state)
    _PARALLEL_RENDER_STATE["renderer"] = None


def render_frames_parallel(visualizer, progress_factory) -> None:
    """
    Render all frames with a process pool.

    Falls back to serial rendering if parallel worker startup fails.
    """
    if visualizer.parallel_render_workers == 1:
        visualizer._render_frames_serial()
        return
    if mp.current_process().name != "MainProcess":
        raise RuntimeError(
            "Parallel rendering must be started from the main process. "
            "Ensure your entry script only calls render code under "
            "if __name__ == '__main__'."
        )

    worker_state = {
        "particles": _make_spawn_safe_render_items(visualizer.particles),
        "vector_field": (
            _make_spawn_safe_render_items(visualizer.vector_field)
            if visualizer.vector_field is not None
            else None
        ),
        "camera": visualizer.camera,
        "view_matrix": getattr(visualizer, "view_matrix", None),
        "frame_folder": visualizer.frame_folder,
        "renderer_resolution": visualizer.renderer_resolution,
        "renderer_spp": visualizer.renderer_spp,
    }

    executor = None
    futures = []
    interrupted = False
    try:
        start_method = "spawn"
        executor = ProcessPoolExecutor(
            max_workers=visualizer.parallel_render_workers,
            mp_context=mp.get_context(start_method),
            initializer=_initialize_parallel_worker,
            initargs=(worker_state,),
        )
        max_pending = max(1, visualizer.parallel_render_workers * 2)
        next_frame_index = 0
        while next_frame_index < min(visualizer.number_of_steps, max_pending):
            futures.append(
                executor.submit(_render_frame_parallel_worker, next_frame_index)
            )
            next_frame_index += 1
        with progress_factory() as progress:
            task = progress.add_task(
                "Saving scenes...", total=visualizer.number_of_steps
            )
            while futures:
                done_future = next(as_completed(futures))
                futures.remove(done_future)
                done_future.result()
                progress.update(task, advance=1)
                if next_frame_index < visualizer.number_of_steps:
                    futures.append(
                        executor.submit(_render_frame_parallel_worker, next_frame_index)
                    )
                    next_frame_index += 1
    except KeyboardInterrupt:
        interrupted = True
        logger.warning(
            "Parallel rendering interrupted by user; cancelling pending tasks."
        )
        for future in futures:
            future.cancel()
        if executor is not None:
            _force_stop_executor_processes(executor)
            executor.shutdown(wait=False, cancel_futures=True)
        _force_stop_active_children()
        raise
    except Exception:
        logger.exception(
            "Parallel rendering failed, likely due to worker initialization "
            "or frame-state serialization. Falling back to serial rendering "
            "for this run."
        )
        visualizer._render_frames_serial()
    finally:
        if executor is not None and not interrupted:
            executor.shutdown(wait=True, cancel_futures=False)
