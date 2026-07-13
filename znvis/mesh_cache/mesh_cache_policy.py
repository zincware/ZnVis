"""
Pure lazy mesh cache policy helpers.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetWindow:
    """
    Explicit cache policy output containing exactly what to keep and what to build.
    """

    retained_frames: set[int]
    build_priority: list[int]


@dataclass(frozen=True)
class MeshCachePolicy:
    """
    Pure cache-policy math for startup fill and rolling refill windows.

    Attributes
    ----------
    future_fraction : float
        Sets how much future and how much past frames are put into cache for
        compatibility for different play directions.
    """

    future_fraction: float

    def __post_init__(self) -> None:
        if not (0 <= self.future_fraction <= 1.0):
            raise ValueError(
                "future_fraction needs to be between 0.00 and 1.0 "
                f"but is {self.future_fraction}."
            )

    def get_target_window(
        self,
        current_frame: int,
        do_rewind: bool,
        frame_step: int,
        number_of_steps: int,
        initial_cached_frame_count: int,
    ) -> TargetWindow:
        """
        This method calculates the exact set of frames to keep in memory and
        the exact ordered list of frames to build, wrapping around trajectory
        boundaries based on the current playback state.

        Parameters
        ----------
        current_frame : int
                The current playback frame index.
        do_rewind : bool
                The current playback direction. True if playing backwards.
        frame_step : int
                The speed multiplier for playback (stride).
        number_of_steps : int
                The total number of frames in the trajectory.
        initial_cached_frame_count : int
                The number of complete frames that fit into the cache budget
                during initialization. Used to bound the rolling window size.

        Returns
        -------
        target_window : TargetWindow
                An explicit data object containing the set of frames to retain
                and the prioritized list of frames to build.
        """
        if number_of_steps <= 0:
            return TargetWindow(set(), [])

        frame_count = number_of_steps
        if initial_cached_frame_count > 0:
            frame_count = min(number_of_steps, initial_cached_frame_count)
        frame_count = max(1, int(frame_count))

        future_count = max(
            1, min(frame_count, int(round(frame_count * self.future_fraction)))
        )
        past_count = frame_count - future_count

        stride = max(1, int(frame_step))
        actual_stride = -stride if do_rewind else stride

        if do_rewind:
            offsets = range(-(future_count - 1), past_count + 1)
        else:
            offsets = range(-past_count, future_count)

        retained_list = [
            (current_frame + (offset * actual_stride)) % number_of_steps
            for offset in offsets
        ]
        retained_set = set(retained_list)

        priority = []

        # Prioritize frames in the direction of playback
        frame_index = current_frame
        for _ in range(len(retained_list)):
            if frame_index in retained_set and frame_index not in priority:
                priority.append(frame_index)
            frame_index = (frame_index + actual_stride) % number_of_steps
            if frame_index == current_frame:
                break

        # Append remaining history frames as lowest priority
        ordered_frames = (
            list(reversed(retained_list)) if do_rewind else list(retained_list)
        )
        for frame_index in ordered_frames:
            if frame_index not in priority:
                priority.append(frame_index)

        return TargetWindow(retained_frames=retained_set, build_priority=priority)
