"""Unit tests for internal parallel rendering helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from znvis.visualizer import _parallel


class _DummyProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_task(self, *_args, **_kwargs):
        return "task"

    def update(self, *_args, **_kwargs):
        return None


class TestParallelRender(unittest.TestCase):
    def _make_visualizer(self, workers=2):
        vis = SimpleNamespace()
        vis.parallel_render_workers = workers
        vis.number_of_steps = 3
        vis.particles = []
        vis.vector_field = None
        vis.camera = None
        vis.view_matrix = [[1, 0, 0, 0]]
        vis.frame_folder = "/tmp"
        vis.renderer_resolution = [64, 64]
        vis.renderer_spp = 1
        vis._render_frames_serial = MagicMock()
        return vis

    def test_workers_one_uses_serial(self):
        visualizer = self._make_visualizer(workers=1)
        _parallel.render_frames_parallel(visualizer, _DummyProgress)
        visualizer._render_frames_serial.assert_called_once()

    @patch("znvis.visualizer._parallel.mp.current_process")
    def test_non_main_process_raises(self, current_process_mock):
        visualizer = self._make_visualizer(workers=2)
        current_process_mock.return_value = SimpleNamespace(name="SpawnProcess-1")

        with self.assertRaises(RuntimeError):
            _parallel.render_frames_parallel(visualizer, _DummyProgress)

    @patch("znvis.visualizer._parallel.mp.current_process")
    @patch("znvis.visualizer._parallel.ProcessPoolExecutor")
    def test_executor_failure_falls_back_to_serial(
        self, executor_mock, current_process_mock
    ):
        visualizer = self._make_visualizer(workers=2)
        current_process_mock.return_value = SimpleNamespace(name="MainProcess")
        executor_mock.side_effect = RuntimeError("boom")

        _parallel.render_frames_parallel(visualizer, _DummyProgress)

        visualizer._render_frames_serial.assert_called_once()

    @patch("znvis.visualizer._parallel._force_stop_active_children")
    @patch("znvis.visualizer._parallel._force_stop_executor_processes")
    @patch("znvis.visualizer._parallel.as_completed")
    @patch("znvis.visualizer._parallel.mp.current_process")
    @patch("znvis.visualizer._parallel.ProcessPoolExecutor")
    def test_keyboard_interrupt_cleans_up_workers(
        self,
        executor_cls_mock,
        current_process_mock,
        as_completed_mock,
        stop_executor_mock,
        stop_children_mock,
    ):
        visualizer = self._make_visualizer(workers=2)
        current_process_mock.return_value = SimpleNamespace(name="MainProcess")

        future = MagicMock()
        future.result.return_value = 0

        executor = MagicMock()
        executor.submit.return_value = future
        executor_cls_mock.return_value = executor

        as_completed_mock.side_effect = KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            _parallel.render_frames_parallel(visualizer, _DummyProgress)

        future.cancel.assert_called()
        stop_executor_mock.assert_called_once_with(executor)
        executor.shutdown.assert_called_with(wait=False, cancel_futures=True)
        stop_children_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
