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
Video management utilities for ZnVis.
"""

import pathlib
import re
from typing import List, Union

import cv2
from rich.progress import track


class VideoManager:
    """
    Professional video creation and management for ZnVis.

    Handles codec selection, frame validation, and video export
    with proper error handling and resource management.
    """

    # Codec mapping for different formats
    CODEC_MAP = {
        "mp4": "mp4v",  # MPEG-4 Part 2 (widely compatible)
        "avi": "XVID",  # Xvid codec for AVI
        "mov": "mp4v",  # QuickTime format with MPEG-4
        "mkv": "XVID",  # Matroska with Xvid
        "wmv": "WMV2",  # Windows Media Video
        "webm": "VP80",  # WebM with VP8
        "flv": "FLV1",  # Flash Video
        "mpg": "MPG1",  # MPEG-1
        "mpeg": "MPG1",  # MPEG-1
        "m4v": "mp4v",  # MPEG-4 Video
        "ogv": "THEO",  # Ogg Video (Theora)
        "3gp": "H263",  # 3GPP format
        "h264": "H264",  # H.264 codec
        "hevc": "HEVC",  # H.265/HEVC codec
        "apng": "apng",  # Animated PNG (not widely supported)
    }

    def __init__(self, output_folder: Union[str, pathlib.Path], frame_rate: int = 60):
        """
        Initialize the VideoManager.

        Parameters
        ----------
        output_folder : Union[str, pathlib.Path]
            Directory where videos will be saved
        frame_rate : int, default=60
            Frame rate for video export
        """
        self.output_folder = pathlib.Path(output_folder).resolve()
        self.frame_rate = frame_rate

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """
        Get list of supported video formats.

        Returns
        -------
        List[str]
            List of supported video format extensions
        """
        return list(cls.CODEC_MAP.keys())

    def get_video_codec(self, video_format: str) -> str:
        """
        Get the appropriate codec for the specified video format.

        Parameters
        ----------
        video_format : str
            The video format extension (e.g., 'mp4', 'avi', 'mov')

        Returns
        -------
        str
            The appropriate fourcc codec string

        Raises
        ------
        ValueError
            If the video format is not supported
        """
        format_lower = video_format.lower()
        if format_lower not in self.CODEC_MAP:
            supported_formats = ", ".join(self.CODEC_MAP.keys())
            raise ValueError(
                f"Unsupported video format '{video_format}'. "
                f"Supported formats: {supported_formats}"
            )

        return self.CODEC_MAP[format_lower]

    def validate_video_format(self, video_format: str) -> str:
        """
        Validate video format and return corrected format if needed.

        Parameters
        ----------
        video_format : str
            The video format to validate

        Returns
        -------
        str
            Valid video format (defaults to 'mp4' if invalid)
        """
        try:
            self.get_video_codec(video_format)
            return video_format
        except ValueError:
            print(
                f"Warning: Invalid video format '{video_format}'. Defaulting to 'mp4'."
            )
            return "mp4"

    def create_video_from_frames(
        self,
        frame_folder: Union[str, pathlib.Path],
        video_name: str = "ZnVis-Video",
        video_format: str = "mp4",
        keep_frames: bool = True,
        frame_pattern: str = "*.png",
    ) -> pathlib.Path:
        """
        Create a video from image frames.

        Parameters
        ----------
        frame_folder : Union[str, pathlib.Path]
            Directory containing the image frames
        video_name : str, default="ZnVis-Video"
            Name of the output video (without extension)
        video_format : str, default="mp4"
            Video format extension
        keep_frames : bool, default=True
            Whether to keep the frame images after video creation
        frame_pattern : str, default="*.png"
            Glob pattern to match frame files

        Returns
        -------
        pathlib.Path
            Path to the created video file

        Raises
        ------
        RuntimeError
            If no frames are found or video creation fails
        """
        frame_folder = pathlib.Path(frame_folder)

        # Find and sort image frames
        images = list(frame_folder.glob(frame_pattern))
        if not images:
            raise RuntimeError(
                f"No frames found in {frame_folder} with pattern '{frame_pattern}'"
            )

        # Sort images by number (assumes numeric naming)

        images = sorted(images, key=lambda s: int(re.search(r"\d+", s.name).group()))

        # Read first frame to get dimensions
        first_image_path = images[0].as_posix()
        single_frame = cv2.imread(first_image_path)
        if single_frame is None:
            raise RuntimeError(f"Could not read first frame: {first_image_path}")
        height, width = single_frame.shape[:2]

        # Validate format and get codec
        video_format = self.validate_video_format(video_format)
        try:
            codec = self.get_video_codec(video_format)
            fourcc = cv2.VideoWriter_fourcc(*codec)
        except ValueError as e:
            print(f"Warning: {e}. Falling back to mp4v codec.")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_format = "mp4"

        # Create output path
        self.output_folder.mkdir(parents=True, exist_ok=True)
        output_path = self.output_folder / f"{video_name}.{video_format}"

        # Create video writer
        video = cv2.VideoWriter(
            output_path.as_posix(),
            fourcc,
            self.frame_rate,
            (width, height),
        )
        if not video.isOpened():
            raise RuntimeError(
                f"Failed to initialize VideoWriter for '{output_path}' "
                f"with codec '{codec} at {width}x{height}@{self.frame_rate}fps."
            )

        # Write frames to video
        try:
            for image_path in track(
                [img.as_posix() for img in images], description="Exporting Video..."
            ):
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"Warning: Could not read frame {image_path}, skipping...")
                    continue
                video.write(frame)
        except Exception as e:
            raise RuntimeError(f"Error while writing video frames: {e}")
        finally:
            video.release()
            cv2.destroyAllWindows()

        print(f"Video successfully created: {output_path}")

        # Clean up frames if requested
        if not keep_frames:
            for img in images:
                try:
                    pathlib.Path(img).unlink(missing_ok=True)
                except Exception as e:
                    print(f"Warning: Failed to delete frame {img}: {e}")
            # Remove folder if empty
            try:
                frame_folder.rmdir()
            except OSError:
                # Directory not empty or cannot be removed; ignore
                pass
            print(f"Temporary frames deleted in: {frame_folder}")
        return output_path

    def get_video_info(self, video_path: Union[str, pathlib.Path]) -> dict:
        """
        Get information about an existing video file.
        This is used in the testing environment.

        Parameters
        ----------
        video_path : Union[str, pathlib.Path]
            Path to the video file

        Returns
        -------
        dict
            Dictionary containing video information (width, height, fps, frame_count)
        """
        video_path = pathlib.Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path.as_posix())
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = (frame_count / fps) if fps > 1e-6 else None
        try:
            info = {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
            }
        finally:
            cap.release()

        return info
