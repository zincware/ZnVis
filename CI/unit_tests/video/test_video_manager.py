import os
import pathlib
import shutil
import unittest

import cv2
import numpy as np

from znvis.video import VideoManager


class TestVideoManager(unittest.TestCase):
    """
    A test class for the VideoManager class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare an instance of the VideoManager class for the test.

        Returns
        -------
        """
        project_root = pathlib.Path(__file__).resolve().parents[2]
        path_to_test_dir = project_root / "test_files" / "video_manager"
        cls.temp_dir = path_to_test_dir
        os.makedirs(cls.temp_dir, exist_ok=True)
        cls.frame_folder = pathlib.Path(cls.temp_dir) / "frames"
        cls.frame_folder.mkdir(parents=True, exist_ok=True)
        cls.output_folder = pathlib.Path(cls.temp_dir) / "output"
        cls.output_folder.mkdir(parents=True, exist_ok=True)
        cls.frame_rate = 10
        cls.manager = VideoManager(
            output_folder=cls.output_folder, frame_rate=cls.frame_rate
        )
        # Dummy frames
        for i in range(10):
            img = np.full((100, 200, 3), i * 25, dtype=np.uint8)
            cv2.imwrite(str(cls.frame_folder / f"frame_{i:03d}.png"), img)

        cls.path_to_bad_dir = project_root / "test_files" / "video_manager" / "bad_dir"
        cls.bad_path_manager = VideoManager(
            output_folder=cls.path_to_bad_dir, frame_rate=cls.frame_rate
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Clean up the test directory after all tests are completed.

        Returns
        -------
        """
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
            print(f"Cleaned up test directory: {cls.temp_dir}")

    def test_codec_selection(self):

        self.assertTrue(self.manager.get_video_codec("mp4") == "mp4v")
        self.assertTrue(self.manager.get_video_codec("avi") == "XVID")
        self.assertTrue(self.manager.get_video_codec("webm") == "VP80")

    def test_format_validation(self):
        self.assertTrue(self.manager.validate_video_format("mp4") == "mp4")
        self.assertTrue(self.manager.validate_video_format("xyz") == "mp4")  # fallback

    def test_create_video_from_frames_mp4(self):
        video_path = self.manager.create_video_from_frames(
            frame_folder=self.frame_folder,
            video_name="test_video",
            video_format="mp4",
            keep_frames=False,
            frame_pattern="*.png",
        )
        self.assertTrue(video_path.exists())
        info = self.manager.get_video_info(video_path)
        self.assertTrue(info["width"], 200)
        self.assertTrue(info["height"] == 100)
        self.assertTrue(abs(info["fps"] - self.frame_rate < 0.5))
        self.assertTrue(info["frame_count"] == 10)

    def test_create_video_from_frames_invalid_format(self):
        video_path = self.manager.create_video_from_frames(
            frame_folder=self.frame_folder,
            video_name="test_video_invalid",
            video_format="xyz",
            keep_frames=True,
            frame_pattern="*.png",
        )
        self.assertTrue(video_path.exists())
        self.assertTrue(str(video_path).endswith(".mp4"))

    def test_get_supported_formats(self):
        supported_formats = [
            "mp4",
            "avi",
            "mov",
            "mkv",
            "wmv",
            "webm",
            "flv",
            "mpg",
            "mpeg",
            "m4v",
            "ogv",
            "3gp",
            "h264",
            "hevc",
            "apng",
        ]
        # Test that get_supported_formats returns the same
        # list as the expected supported_formats
        actual_formats = self.manager.get_supported_formats()
        self.assertEqual(sorted(actual_formats), sorted(supported_formats))

    def test_bad_path(self):
        with self.assertRaises((RuntimeError)) as context:
            self.bad_path_manager.create_video_from_frames(
                frame_folder=self.path_to_bad_dir,
                video_name="this_should_not_work",
                video_format="mov",
                keep_frames=False,
                frame_pattern="*.png",
            )
        # Check if error message is correct
        self.assertTrue(str(context.exception).startswith("No frames found in "))
