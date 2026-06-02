"""
Lazy mesh cache policy and background refill management.
"""

import threading
from collections.abc import Iterable

from rich.progress import Progress

from znvis.visualizer.cache.mesh_cache_policy import MeshCachePolicy, TargetWindow
from znvis.visualizer.cache.mesh_frame_cache import MeshFrameCache


class MeshCacheManager:
    """
    Manage lazy mesh cache policy and its single background refill worker.
    """

    def __init__(
        self,
        items: list,
        number_of_steps: int,
        cache: MeshFrameCache,
        future_fraction: float = 2 / 3,
        chunk_size: int = 10,
    ):
        """

        Parameters
        ----
        chunk_size : int
            The maximum number of frames the background worker is allowed to build in
            a single iteration. By batching the workload into these smaller chunks,
            the worker thread regularly yields execution. This ensures it remains
            highly responsive to sudden changes in user playback (like reversing or
            changing speed) without getting locked into long, blocking render loops.
        """
        self.items = items
        self.number_of_steps = number_of_steps
        self.cache = cache
        self.policy = MeshCachePolicy(future_fraction=future_fraction)

        self.chunk_size = chunk_size

        self.complete_frames: set[int] = set()
        self._map_lock = threading.Lock()

        self.initial_cached_frame_count = 0
        self.initial_loaded = False
        self.current_frame = 0
        self.do_rewind = False
        self.frame_step = 1
        self.urgent_frame = None
        self.use_pause_buffer = False
        self._cache_refill_stop_event = threading.Event()
        self._cache_manager_stop_event = threading.Event()
        self._cache_manager_wake_event = threading.Event()
        self._cache_manager_thread = None

    @property
    def static_items(self) -> list:
        return [item for item in self.items if item.static]

    @property
    def dynamic_items(self) -> list:
        return [item for item in self.items if not item.static]

    def get(self, item, frame_index: int):
        """
        Return a cached mesh, building it if necessary.
        """
        idx = 0 if item.static else frame_index
        return self.cache.get(item, idx)

    def initialize(
        self,
        current_frame: int,
        do_rewind: bool,
        frame_step: int,
        start_worker: bool = False,
    ) -> None:
        """
        Fill the initial balanced cache window and optionally start the worker.
        """
        self.update_playback_state(current_frame, do_rewind, frame_step)
        if self.initial_loaded:
            self.ensure_current_frame(current_frame)
            self.submit_prefetch(current_frame, do_rewind, frame_step)
            return

        target_window: TargetWindow = self.policy.get_target_window(
            current_frame=current_frame,
            do_rewind=do_rewind,
            frame_step=frame_step,
            number_of_steps=self.number_of_steps,
            initial_cached_frame_count=0,  # 0 forces it to target the full trajectory
        )

        total = len(self.static_items) + len(self.dynamic_items) * self.number_of_steps
        with Progress() as progress:
            task = progress.add_task("Preparing mesh cache...", total=total)
            completed, budget_limited = self.ensure_mesh_window(
                frames=target_window.build_priority,
                progress=progress,
                task=task,
                allow_eviction=False,
            )
            if budget_limited:
                progress.update(
                    task,
                    completed=completed,
                    total=completed,
                    description="Mesh cache filled",
                )

        self.initial_loaded = True
        self.initial_cached_frame_count = len(self.complete_frames)
        if start_worker:
            self.start()

    def update_playback_state(
        self,
        current_frame: int,
        do_rewind: bool,
        frame_step: int,
    ) -> None:
        """
        Store current playback state for asynchronous refill iterations.
        """
        self.current_frame = current_frame
        self.do_rewind = do_rewind
        self.frame_step = max(1, int(frame_step))

    def ensure_mesh_window(
        self,
        frames: Iterable[int],
        progress=None,
        task=None,
        stop_event: threading.Event | None = None,
        allow_eviction: bool = True,
    ) -> tuple[int, bool]:
        """
        Build missing meshes for ``frames`` until stopped or budget-limited.
        Maintains the complete_frames set.
        """
        completed = 0

        for item in self.static_items:
            result = self.cache.get_with_status(item, 0)
            if result.created:
                completed += 1
                if progress is not None:
                    progress.update(task, advance=1)

        for frame_index in frames:
            if stop_event is not None and stop_event.is_set():
                return completed, False

            frame_is_complete = True

            for item in self.dynamic_items:
                if stop_event is not None and stop_event.is_set():
                    return completed, False

                result = self.cache.get_with_status(
                    item,
                    frame_index,
                    allow_eviction=allow_eviction,
                )

                if result.evicted_frames:
                    self.complete_frames.difference_update(result.evicted_frames)

                if result.created:
                    completed += 1
                    if progress is not None:
                        progress.update(task, advance=1)

                if not result.stored:
                    frame_is_complete = False
                    evicted = self.cache.evict_dynamic_frame(frame_index)
                    with self._map_lock:
                        self.complete_frames.difference_update(evicted)
                    return completed, True

            if frame_is_complete:
                with self._map_lock:
                    self.complete_frames.add(frame_index)

        return completed, False

    def ensure_current_frame(self, frame_index: int) -> None:
        """
        Synchronously ensure the display frame is cached.
        """
        frame_is_complete = True
        for item in self.items:
            idx = 0 if item.static else frame_index
            result = self.cache.get_with_status(item, idx)
            if result.evicted_frames:
                with self._map_lock:
                    self.complete_frames.difference_update(result.evicted_frames)
            if not item.static and not result.stored:
                frame_is_complete = False

        if frame_is_complete:
            with self._map_lock:
                self.complete_frames.add(frame_index)

    def has_meshes_for_frame(self, frame_index: int) -> bool:
        """
        Check if all meshes needed for ``frame_index`` are cached.
        """
        with self._map_lock:
            return frame_index in self.complete_frames

    def get_complete_frames_count(self) -> int:
        """Thread-safe way to check how many frames are complete."""
        with self._map_lock:
            return len(self.complete_frames)

    def update_once(self) -> tuple[int, bool]:
        """
        Move the cache toward the current policy target window once.
        """
        if not self.use_pause_buffer:
            self._cache_refill_stop_event.clear()

        current_capacity = self.get_complete_frames_count()
        desired_window = min(
            self.initial_cached_frame_count, current_capacity + self.chunk_size
        )

        target: TargetWindow = self.policy.get_target_window(
            current_frame=self.current_frame,
            do_rewind=self.do_rewind,
            frame_step=self.frame_step,
            number_of_steps=self.number_of_steps,
            initial_cached_frame_count=desired_window,
        )

        # Evict out-of-bounds frames and update the map
        evicted = self.cache.evict_dynamic_frames_not_in(target.retained_frames)
        with self._map_lock:
            self.complete_frames.difference_update(evicted)

        # Compute the chunk once using the complete map
        batch = []
        if self.urgent_frame is not None and not self.has_meshes_for_frame(
            self.urgent_frame
        ):
            batch.append(self.urgent_frame)
            self.urgent_frame = None

        for f in target.build_priority:
            if len(batch) >= self.chunk_size:
                break
            if f not in self.complete_frames and f not in batch:
                batch.append(f)

        if not batch:
            return 0, False

        # Build the chunk
        completed_count, budget_limited = self.ensure_mesh_window(
            frames=batch,
            stop_event=self._cache_refill_stop_event,
        )

        # If a full chunk is built without hitting limits, there is likely more work
        if len(batch) == self.chunk_size and not budget_limited:
            self.wake()

        return completed_count, budget_limited

    def start(self) -> None:
        """
        Start the persistent background cache manager thread.
        """
        if (
            self._cache_manager_thread is not None
            and self._cache_manager_thread.is_alive()
        ):
            return
        self._cache_manager_stop_event.clear()
        self._cache_manager_thread = threading.Thread(
            target=self.run,
            daemon=True,
        )
        self._cache_manager_thread.start()

    def run(self) -> None:
        """
        Process wake events until shutdown.
        """
        while not self._cache_manager_stop_event.is_set():
            self.run_once(timeout=None)

    def run_once(self, timeout: float | None = None) -> None:
        """
        Run one manager iteration only after a wake event.
        """
        was_woken = self._cache_manager_wake_event.wait(timeout=timeout)
        if not was_woken:
            return
        self._cache_manager_wake_event.clear()
        if self._cache_manager_stop_event.is_set():
            return
        self.update_once()

    def wake(self) -> None:
        """
        Ask the worker to re-evaluate the cache policy.
        """
        self._cache_manager_wake_event.set()

    def submit_prefetch(
        self,
        current_frame: int,
        do_rewind: bool,
        frame_step: int,
        urgent_frame: int | None = None,
    ) -> None:
        """
        Wake the worker for active playback refill.
        """
        self.use_pause_buffer = False
        self._cache_refill_stop_event.set()
        self.update_playback_state(current_frame, do_rewind, frame_step)
        if urgent_frame is not None:
            self.urgent_frame = urgent_frame
        self.wake()

    def submit_pause_refill(
        self,
        current_frame: int,
        do_rewind: bool,
        frame_step: int,
    ) -> None:
        """
        Wake the worker for paused refill using the same cache policy.
        """
        self.use_pause_buffer = True
        self.update_playback_state(current_frame, do_rewind, frame_step)
        self.wake()

    def cancel_pause_refill(self) -> None:
        """
        Stop the current refill attempt.
        """
        self.use_pause_buffer = False
        self._cache_refill_stop_event.set()

    def shutdown(self) -> None:
        """
        Stop the background worker.
        """
        self._cache_manager_stop_event.set()
        self._cache_refill_stop_event.set()
        self._cache_manager_wake_event.set()
        if self._cache_manager_thread is not None:
            self._cache_manager_thread.join(timeout=1)

    def get_cached_frame_ranges(self) -> str:
        """
        Return compact complete-frame ranges for debug output.
        """
        ranges = []
        start = None
        previous = None
        for frame_index in range(self.number_of_steps):
            if self.has_meshes_for_frame(frame_index):
                if start is None:
                    start = frame_index
                previous = frame_index
                continue
            if start is not None:
                ranges.append((start, previous))
                start = None
                previous = None
        if start is not None:
            ranges.append((start, previous))

        if not ranges:
            return "none"
        return ",".join(
            str(first) if first == last else f"{first}-{last}" for first, last in ranges
        )
