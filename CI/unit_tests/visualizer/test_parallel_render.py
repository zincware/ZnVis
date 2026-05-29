"""Unit tests for internal parallel rendering helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from znvis.visualizer import _parallel_render
from znvis.visualizer import base_visualizer as _parallel_base


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
    @patch.dict("os.environ", {}, clear=True)
    @patch("sys.modules", {"drjit": None})
    def test_gpu_detection_does_not_probe_mitsuba_variant(self):
        with patch("mitsuba.set_variant") as set_variant_mock:
            gpu_count = _parallel_base._detect_available_gpu_devices()

        self.assertEqual(gpu_count, 0)
        set_variant_mock.assert_not_called()

    def _make_visualizer(self, workers=2, gpus=1):
        vis = SimpleNamespace()
        vis.parallel_render_workers = workers
        vis.available_gpu_devices = gpus
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

    def test_worker_specs_use_one_worker_per_visible_gpu(self):
        visualizer = self._make_visualizer(workers=2, gpus=2)

        specs = _parallel_render._build_worker_specs(visualizer)

        self.assertEqual([spec.gpu_id for spec in specs], [0, 1])

    def test_worker_specs_distribute_extra_workers_round_robin_and_warn(self):
        visualizer = self._make_visualizer(workers=5, gpus=2)

        with self.assertWarnsRegex(UserWarning, "experimental"):
            specs = _parallel_render._build_worker_specs(visualizer)

        self.assertEqual([spec.gpu_id for spec in specs], [0, 1, 0, 1, 0])

    def test_worker_specs_raise_without_visible_gpu(self):
        visualizer = self._make_visualizer(workers=2, gpus=0)

        with self.assertRaisesRegex(RuntimeError, "CUDA GPU"):
            _parallel_render._build_worker_specs(visualizer)

    @patch("znvis.visualizer._parallel_render.subprocess.Popen")
    def test_start_worker_process_pins_cuda_visible_device(self, popen_mock):
        spec = _parallel_render._WorkerSpec(gpu_id=1, cuda_visible_device="3")

        _parallel_render._start_worker_process(spec, "/tmp/state.pkl")

        env = popen_mock.call_args.kwargs["env"]
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")

    def test_parallel_without_visible_gpu_falls_back_to_serial_and_warns(self):
        visualizer = self._make_visualizer(workers=2, gpus=0)

        with self.assertWarnsRegex(UserWarning, "falling back to serial"):
            _parallel_render.render_frames_parallel(visualizer, _DummyProgress)

        visualizer._render_frames_serial.assert_called_once()

    def test_workers_one_uses_serial_shortcut(self):
        visualizer = self._make_visualizer(workers=1, gpus=1)
        _parallel_render.render_frames_parallel(visualizer, _DummyProgress)
        visualizer._render_frames_serial.assert_called_once()

    @patch("znvis.visualizer._parallel_render.mp.current_process")
    def test_non_main_process_raises(self, current_process_mock):
        visualizer = self._make_visualizer(workers=2, gpus=2)
        current_process_mock.return_value = SimpleNamespace(name="SpawnProcess-1")

        with self.assertRaises(RuntimeError):
            _parallel_render.render_frames_parallel(visualizer, _DummyProgress)

    @patch("znvis.visualizer._parallel_render.mp.current_process")
    @patch("znvis.visualizer._parallel_render._start_worker_process")
    def test_worker_start_failure_falls_back_to_serial(
        self, start_worker_mock, current_process_mock
    ):
        visualizer = self._make_visualizer(workers=2, gpus=2)
        current_process_mock.return_value = SimpleNamespace(name="MainProcess")
        start_worker_mock.side_effect = RuntimeError("boom")

        _parallel_render.render_frames_parallel(visualizer, _DummyProgress)

        visualizer._render_frames_serial.assert_called_once()

    @patch("znvis.visualizer._parallel_render.selectors.DefaultSelector")
    @patch("znvis.visualizer._parallel_render._force_stop_processes")
    @patch("znvis.visualizer._parallel_render.mp.current_process")
    @patch("znvis.visualizer._parallel_render._start_worker_process")
    def test_keyboard_interrupt_cleans_up_workers(
        self,
        start_worker_mock,
        current_process_mock,
        stop_processes_mock,
        selector_cls_mock,
    ):
        visualizer = self._make_visualizer(workers=2, gpus=2)
        current_process_mock.return_value = SimpleNamespace(name="MainProcess")

        process = MagicMock()
        process.stdout = MagicMock()
        process.poll.return_value = None
        start_worker_mock.return_value = process

        selector = MagicMock()
        selector.register.side_effect = [None, KeyboardInterrupt]
        selector_cls_mock.return_value = selector

        with self.assertRaises(KeyboardInterrupt):
            _parallel_render.render_frames_parallel(visualizer, _DummyProgress)

        stop_processes_mock.assert_called_once()
        self.assertEqual(stop_processes_mock.call_args.args[0], [process, process])


if __name__ == "__main__":
    unittest.main()
