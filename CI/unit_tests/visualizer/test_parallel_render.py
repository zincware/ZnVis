"""
ZnVis: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0
Copyright Contributors to the Zincwarecode Project.
Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
Unit tests for internal parallel rendering helpers.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import znvis.parallel_render.parallel_render_manager as parallel_render_manager


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
    def _make_render_config(self, workers=2, gpus=1):
        """Hilfsmethode, um ein passendes render_config-Dictionary zu erzeugen."""
        return {
            "number_of_steps": 3,
            "parallel_render_workers": workers,
            "available_gpu_devices": gpus,
            "worker_state": {
                "particles": [],
                "vector_field": None,
                "camera": None,
                "view_matrix": [[1, 0, 0, 0]],
                "frame_folder": "/tmp",
                "renderer_resolution": [64, 64],
                "renderer_spp": 1,
            },
        }

    def test_worker_specs_use_one_worker_per_visible_gpu(self):
        config = self._make_render_config(workers=2, gpus=2)

        specs = parallel_render_manager._build_worker_specs(
            config["parallel_render_workers"], config["available_gpu_devices"]
        )

        self.assertEqual([spec.gpu_id for spec in specs], [0, 1])

    def test_worker_specs_distribute_extra_workers_round_robin_and_warn(self):
        config = self._make_render_config(workers=5, gpus=2)

        with self.assertWarnsRegex(UserWarning, "experimental"):
            specs = parallel_render_manager._build_worker_specs(
                config["parallel_render_workers"], config["available_gpu_devices"]
            )

        self.assertEqual([spec.gpu_id for spec in specs], [0, 1, 0, 1, 0])

    def test_worker_specs_raise_without_visible_gpu(self):
        config = self._make_render_config(workers=2, gpus=0)

        with self.assertRaisesRegex(RuntimeError, "CUDA GPU"):
            parallel_render_manager._build_worker_specs(
                config["parallel_render_workers"], config["available_gpu_devices"]
            )

    @patch("znvis.parallel_render.parallel_render_manager.subprocess.Popen")
    def test_start_worker_process_pins_cuda_visible_device(self, popen_mock):
        spec = parallel_render_manager._WorkerSpec(gpu_id=1, cuda_visible_device="3")

        parallel_render_manager._start_worker_process(spec, "/tmp/state.pkl")

        env = popen_mock.call_args.kwargs["env"]
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")

    def test_parallel_without_visible_gpu_raises_runtime_error(self):
        config = self._make_render_config(workers=2, gpus=0)

        with self.assertRaisesRegex(
            RuntimeError, "Parallel rendering requirements failed"
        ):
            parallel_render_manager.render_frames_parallel(config, _DummyProgress)

    def test_workers_one_raises_value_error(self):
        config = self._make_render_config(workers=1, gpus=1)

        with self.assertRaisesRegex(
            ValueError, "Parallel rendering invoked with only 1 worker"
        ):
            parallel_render_manager.render_frames_parallel(config, _DummyProgress)

    @patch("znvis.parallel_render.parallel_render_manager.mp.current_process")
    def test_non_main_process_raises(self, current_process_mock):
        config = self._make_render_config(workers=2, gpus=2)
        current_process_mock.return_value = SimpleNamespace(name="SpawnProcess-1")

        with self.assertRaises(RuntimeError):
            parallel_render_manager.render_frames_parallel(config, _DummyProgress)

    @patch("znvis.parallel_render.parallel_render_manager.mp.current_process")
    @patch("znvis.parallel_render.parallel_render_manager._start_worker_process")
    def test_worker_start_failure_bubbles_up_exception(
        self, start_worker_mock, current_process_mock
    ):
        config = self._make_render_config(workers=2, gpus=2)
        current_process_mock.return_value = SimpleNamespace(name="MainProcess")
        start_worker_mock.side_effect = RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            parallel_render_manager.render_frames_parallel(config, _DummyProgress)

    @patch("znvis.parallel_render.parallel_render_manager.selectors.DefaultSelector")
    @patch("znvis.parallel_render.parallel_render_manager._force_stop_processes")
    @patch("znvis.parallel_render.parallel_render_manager.mp.current_process")
    @patch("znvis.parallel_render.parallel_render_manager._start_worker_process")
    def test_keyboard_interrupt_cleans_up_workers(
        self,
        start_worker_mock,
        current_process_mock,
        stop_processes_mock,
        selector_cls_mock,
    ):
        config = self._make_render_config(workers=2, gpus=2)
        current_process_mock.return_value = SimpleNamespace(name="MainProcess")

        process = MagicMock()
        process.stdout = MagicMock()
        process.poll.return_value = None
        start_worker_mock.return_value = process

        selector = MagicMock()
        selector.register.side_effect = [None, KeyboardInterrupt]
        selector_cls_mock.return_value = selector

        with self.assertRaises(KeyboardInterrupt):
            parallel_render_manager.render_frames_parallel(config, _DummyProgress)

        stop_processes_mock.assert_called_once()
        self.assertEqual(stop_processes_mock.call_args.args[0], [process, process])


if __name__ == "__main__":
    unittest.main()
