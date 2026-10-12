"""MP4 recorder with a live, grounded high-level plan header."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio

GroundPlan = list[tuple[str, tuple[str, ...]]]


class PlanVideoRecorder:
    """Record rendered frames while visualizing plan execution state."""

    HEADER_HEIGHT = 112

    def __init__(
        self,
        path: Path,
        plan: GroundPlan,
        fps: int,
        stride: int,
        frame_scale: float = 1.0,
    ) -> None:
        assert fps > 0
        assert stride > 0
        assert frame_scale >= 1.0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.plan = plan
        self.columns = min(5, max(1, len(plan)))
        rows = (len(plan) + self.columns - 1) // self.columns
        self.HEADER_HEIGHT = ((34 + 78 * max(1, rows) + 15) // 16) * 16
        self.fps = fps
        self.stride = stride
        self.frame_scale = frame_scale
        self._pending_frame: Any | None = None
        self._extra_frame_accumulator = 0.0
        self.step = 0
        self.active_index: int | None = None
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264")

    def set_plan(self, plan: GroundPlan, active_skill: str | None = None) -> None:
        """Replace the displayed plan, for example after reactive replanning."""
        self.plan = plan
        self.active_index = None
        self._set_active(active_skill)

    def mark_complete(self) -> None:
        """Display every element of the current plan as completed."""
        self.active_index = len(self.plan)

    def _set_active(self, skill_name: str | None) -> None:
        if skill_name is None:
            self.active_index = None
            return
        start = self.active_index if self.active_index is not None else 0
        for idx in range(start, len(self.plan)):
            if self.plan[idx][0] == skill_name:
                self.active_index = idx
                return

    def _annotate(self, frame: Any, status: str) -> Any:
        frame = frame[..., :3].copy()
        width = frame.shape[1]
        canvas = cv2.copyMakeBorder(
            frame,
            self.HEADER_HEIGHT,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(15, 23, 42),
        )
        cv2.putText(
            canvas,
            status,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (226, 232, 240),
            1,
            cv2.LINE_AA,
        )
        if not self.plan:
            cv2.putText(
                canvas,
                "PLAN: <empty>",
                (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (248, 113, 113),
                2,
                cv2.LINE_AA,
            )
            return canvas

        gap = 8
        left = 10
        box_width = max(
            1,
            (width - 2 * left - gap * (self.columns - 1)) // self.columns,
        )
        for idx, (name, parameters) in enumerate(self.plan):
            row, column = divmod(idx, self.columns)
            dy = 78 * row
            x0 = left + column * (box_width + gap)
            x1 = min(width - left, x0 + box_width)
            if self.active_index is not None and idx < self.active_index:
                color = (22, 101, 52)
                state = "DONE"
            elif idx == self.active_index:
                color = (202, 138, 4)
                state = "ACTIVE"
            else:
                color = (51, 65, 85)
                state = "NEXT"
            cv2.rectangle(canvas, (x0, 34 + dy), (x1, 103 + dy), color, -1)
            cv2.rectangle(canvas, (x0, 34 + dy), (x1, 103 + dy), (148, 163, 184), 1)
            font_scale = min(0.58, max(0.34, box_width / max(170.0, len(name) * 19.0)))
            cv2.putText(
                canvas,
                name,
                (x0 + 6, 58 + dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            args = ", ".join(parameters)
            cv2.putText(
                canvas,
                args,
                (x0 + 6, 79 + dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                min(0.39, max(0.27, box_width / max(260.0, len(args) * 17.0))),
                (226, 232, 240),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                state,
                (x0 + 6, 97 + dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (241, 245, 249),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def capture(
        self,
        system: Any,
        skill_name: str | None,
        status: str,
        *,
        force: bool = False,
    ) -> None:
        """Capture a frame and update the active high-level skill."""
        self._set_active(skill_name)
        if force or self.step % self.stride == 0:
            frame = system.env.render()
            if frame is not None:
                annotated = self._annotate(frame, status)
                if self._pending_frame is None:
                    self._pending_frame = annotated
                else:
                    previous = self._pending_frame
                    self.writer.append_data(previous)
                    self._extra_frame_accumulator += self.frame_scale - 1.0
                    extra_frames = int(self._extra_frame_accumulator + 1e-9)
                    self._extra_frame_accumulator -= extra_frames
                    for index in range(extra_frames):
                        alpha = (index + 1) / (extra_frames + 1)
                        self.writer.append_data(
                            cv2.addWeighted(
                                previous, 1.0 - alpha, annotated, alpha, 0.0
                            )
                        )
                    self._pending_frame = annotated
        self.step += 1

    def hold_final(self, system: Any, status: str, frames: int | None = None) -> None:
        """Append repeated terminal frames so the outcome is easy to inspect."""
        frame = system.env.render()
        if frame is None:
            return
        annotated = self._annotate(frame, status)
        if self._pending_frame is not None:
            self.writer.append_data(self._pending_frame)
            self._pending_frame = None
        if frames is None:
            frames = round(1.2 * self.fps * self.frame_scale)
        for _ in range(frames):
            self.writer.append_data(annotated)

    def close(self) -> None:
        if self._pending_frame is not None:
            self.writer.append_data(self._pending_frame)
            self._pending_frame = None
        self.writer.close()
