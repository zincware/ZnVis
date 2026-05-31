"""
Class for Cache handling of the created meshes for visualized frames.
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

import znvis


@dataclass(frozen=True)
class MeshCacheResult:
    """
    Result of a mesh cache lookup or insert.
    """

    mesh: object
    created: bool
    evicted_frames: set[int] = field(default_factory=set)
    build_seconds: float = 0.0
    stored: bool = True

    @property
    def evicted(self) -> bool:
        """
        Return whether inserting this mesh evicted cached dynamic entries.
        """
        return len(self.evicted_frames) > 0


def estimate_mesh_nbytes(mesh) -> int:
    """
    Estimate raw array storage used by an Open3D triangle mesh.
    """
    total = 0
    for attr in (
        "vertices",
        "triangles",
        "vertex_normals",
        "triangle_normals",
        "vertex_colors",
        "triangle_uvs",
    ):
        values = getattr(mesh, attr, None)
        if values is not None:
            total += np.asarray(values).nbytes
    return int(total)


class MeshFrameCache:
    """
    Class to manage the mesh frame caching in LRU-style.
    """

    def __init__(self, max_bytes: int | None = None):
        """
        Constructor for the visualizer.

         Parameters
         ----------
         max_bytes : int | None
             Maximum total bytes for static and dynamic cache entries.
        """
        self.max_bytes = max_bytes
        self._static_cache = {}
        self._dynamic_cache = OrderedDict()
        self.static_bytes = 0
        self.dynamic_bytes = 0
        self._lock = threading.Lock()

    @property
    def current_bytes(self) -> int:
        """
        Return the estimated total memory used by cached meshes.
        """
        return self.static_bytes + self.dynamic_bytes

    def _evict_dynamic_entries(self) -> set[int]:
        """
        Evict least-recently-used dynamic entries until cache limits are satisfied.
        """
        evicted_frames = set()
        while (
            self._dynamic_cache
            and self.max_bytes is not None
            and self.current_bytes > self.max_bytes
        ):
            key, (_, nbytes) = self._dynamic_cache.popitem(last=False)
            self.dynamic_bytes -= nbytes
            _, frame_index = key
            evicted_frames.add(frame_index)
        return evicted_frames

    def get(self, item: znvis.Particle | znvis.VectorField, frame_index: int):
        """
        Return the mesh for ``item`` at ``frame_index``.
        """
        return self.get_with_status(item, frame_index).mesh

    def get_with_status(
        self,
        item: znvis.Particle | znvis.VectorField,
        frame_index: int,
        allow_eviction: bool = True,
    ) -> MeshCacheResult:
        """
        Return the mesh for ``item`` at ``frame_index``.

        If the mesh is already cached, it is returned and marked as recently used.
        Otherwise, the mesh is built with ``item.get_mesh_for_frame()``, stored in
        the cache, and then returned. Old cache entries are evicted when the cache
        exceeds ``max_bytes`` unless ``allow_eviction`` is false.

        Parameters
        ----
        item : znvis.Particle | znvis.VectorField
            Object in a scene.
        frame_index : int
        """
        is_static = bool(item.static)
        key = id(item) if is_static else (id(item), frame_index)

        with self._lock:
            if is_static and key in self._static_cache:
                return MeshCacheResult(
                    mesh=self._static_cache[key][0],
                    created=False,
                )
            if not is_static and key in self._dynamic_cache:
                self._dynamic_cache.move_to_end(key)
                return MeshCacheResult(
                    mesh=self._dynamic_cache[key][0],
                    created=False,
                )

        start = time.perf_counter()
        mesh = item.get_mesh_for_frame(frame_index)
        build_seconds = time.perf_counter() - start
        nbytes = estimate_mesh_nbytes(mesh)
        evicted_frames = set()

        with self._lock:
            if is_static:
                previous = self._static_cache.get(key)
                if previous is not None:
                    self.static_bytes -= previous[1]
                self._static_cache[key] = (mesh, nbytes)
                self.static_bytes += nbytes
            else:
                previous = self._dynamic_cache.get(key)
                previous_nbytes = previous[1] if previous is not None else 0
                would_exceed = (
                    self.max_bytes is not None
                    and self.current_bytes - previous_nbytes + nbytes > self.max_bytes
                )
                if would_exceed and not allow_eviction:
                    return MeshCacheResult(
                        mesh=mesh,
                        created=False,
                        build_seconds=build_seconds,
                        stored=False,
                    )
                if previous is not None:
                    self.dynamic_bytes -= previous[1]
                self._dynamic_cache[key] = (mesh, nbytes)
                self._dynamic_cache.move_to_end(key)
                self.dynamic_bytes += nbytes
                evicted_frames = self._evict_dynamic_entries()

        return MeshCacheResult(
            mesh=mesh,
            created=True,
            evicted_frames=evicted_frames,
            build_seconds=build_seconds,
        )

    def evict_dynamic_frame(self, frame_index: int) -> set[int]:
        """
        Evict dynamic entries for exactly ``frame_index``.
        """
        evicted_frames = set()
        with self._lock:
            for key in list(self._dynamic_cache.keys()):
                _, key_frame_index = key
                if key_frame_index != frame_index:
                    continue
                _, nbytes = self._dynamic_cache.pop(key)
                self.dynamic_bytes -= nbytes
                evicted_frames.add(frame_index)
        return evicted_frames

    def evict_dynamic_frames_not_in(self, retained_frames: set[int]) -> set[int]:
        """
        Evict dynamic entries whose frame index is outside ``retained_frames``.
        """
        evicted_frames = set()
        with self._lock:
            for key in list(self._dynamic_cache.keys()):
                _, frame_index = key
                if frame_index in retained_frames:
                    continue
                _, nbytes = self._dynamic_cache.pop(key)
                self.dynamic_bytes -= nbytes
                evicted_frames.add(frame_index)
        return evicted_frames

    def contains(
        self, item: znvis.Particle | znvis.VectorField, frame_index: int
    ) -> bool:
        """
        Return whether the mesh for ``item`` at ``frame_index`` is currently cached.
        """
        key = id(item) if item.static else (id(item), frame_index)
        with self._lock:
            if item.static:
                return key in self._static_cache
            return key in self._dynamic_cache
