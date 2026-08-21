from __future__ import annotations

import base64
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ClickMark:
    x: int
    y: int
    label: str = ""


@dataclass
class _DisplayUpdate:
    frame: np.ndarray
    stage: str
    hint: str
    marks: Tuple[ClickMark, ...]


class SolverDisplay:
    """
    运行状态展示窗口。

    - 显示当前正在判定/监控的截图；
    - 点击动作前，在对应截图位置画红框；
    - 窗口顶部覆盖当前阶段与提示；
    - Tk 窗口运行在独立线程，不阻塞求解/ADB；
    - 如果当前 Python 环境没有 tkinter/GUI，自动降级为无窗口模式，
      不影响求解器继续运行。

    不依赖 Pillow。截图通过 OpenCV 编码成 PNG，再交给 tkinter.PhotoImage。
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_width: int = 620,
        max_height: int = 980,
        title: str = "Game Solver 状态窗口",
    ) -> None:
        self.enabled = bool(enabled)
        self.max_width = max(320, int(max_width))
        self.max_height = max(480, int(max_height))
        self.title = title

        self._queue: "queue.Queue[Optional[_DisplayUpdate]]" = queue.Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._failed_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.enabled and self._failed_reason is None

    def start(self) -> None:
        if not self.enabled or self._started:
            return

        self._started = True
        self._thread = threading.Thread(
            target=self._ui_thread,
            name="solver-display",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if not self._started:
            return

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

    def show(
        self,
        frame_bgr: np.ndarray,
        *,
        stage: str,
        hint: str = "",
        marks: Sequence[ClickMark] = (),
    ) -> None:
        if not self.enabled or frame_bgr is None or frame_bgr.size == 0:
            return

        self.start()

        update = _DisplayUpdate(
            frame=frame_bgr.copy(),
            stage=str(stage),
            hint=str(hint),
            marks=tuple(marks),
        )

        # UI 永远只需要“最新状态”。队列满时丢弃旧帧，避免窗口越看越落后。
        while True:
            try:
                self._queue.put_nowait(update)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def show_path(
        self,
        image_path: Path,
        *,
        stage: str,
        hint: str = "",
        marks: Sequence[ClickMark] = (),
    ) -> None:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is not None:
            self.show(frame, stage=stage, hint=hint, marks=marks)

    @staticmethod
    def _draw_marks(
        frame: np.ndarray,
        marks: Sequence[ClickMark],
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        sx = w / 940.0
        sy = h / 2048.0

        half_w = max(24, int(round(58 * sx)))
        half_h = max(42, int(round(92 * sy)))
        thickness = max(2, int(round(4 * (sx + sy) / 2.0)))

        for mark in marks:
            x = int(round(mark.x))
            y = int(round(mark.y))
            x1 = max(0, x - half_w)
            x2 = min(w - 1, x + half_w)
            y1 = max(0, y - half_h)
            y2 = min(h - 1, y + half_h)

            cv2.rectangle(
                out,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                thickness,
                cv2.LINE_AA,
            )
            cv2.circle(
                out,
                (max(0, min(w - 1, x)), max(0, min(h - 1, y))),
                max(5, int(round(8 * (sx + sy) / 2.0))),
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )

            # 标签只用 ASCII/数字，避免 OpenCV putText 中文乱码；
            # 中文阶段/提示由 tkinter 直接绘制。
            if mark.label:
                cv2.putText(
                    out,
                    mark.label,
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.55, 0.75 * sx),
                    (0, 0, 255),
                    max(1, thickness - 1),
                    cv2.LINE_AA,
                )

        return out

    def _ui_thread(self) -> None:
        try:
            import tkinter as tk
        except Exception as exc:
            self._failed_reason = f"无法导入 tkinter: {exc}"
            print(f"展示窗口不可用，已自动关闭窗口功能：{self._failed_reason}")
            return

        try:
            root = tk.Tk()
            root.title(self.title)
            root.configure(bg="#111827")
            root.resizable(True, True)

            canvas = tk.Canvas(
                root,
                bg="#111827",
                highlightthickness=0,
                bd=0,
            )
            canvas.pack(fill="both", expand=True)

            state = {
                "photo": None,
                "closed": False,
            }

            def process_updates() -> None:
                if state["closed"]:
                    return

                latest: Optional[_DisplayUpdate] = None
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break

                    if item is None:
                        state["closed"] = True
                        try:
                            root.destroy()
                        except Exception:
                            pass
                        return

                    latest = item

                if latest is not None:
                    annotated = self._draw_marks(latest.frame, latest.marks)
                    src_h, src_w = annotated.shape[:2]

                    status_h = 116
                    usable_h = max(240, self.max_height - status_h)
                    scale = min(
                        self.max_width / max(1.0, float(src_w)),
                        usable_h / max(1.0, float(src_h)),
                        1.0,
                    )

                    disp_w = max(1, int(round(src_w * scale)))
                    disp_h = max(1, int(round(src_h * scale)))

                    if (disp_w, disp_h) != (src_w, src_h):
                        shown = cv2.resize(
                            annotated,
                            (disp_w, disp_h),
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        shown = annotated

                    ok, encoded = cv2.imencode(".png", shown)
                    if ok:
                        png_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
                        photo = tk.PhotoImage(data=png_b64, format="png")
                        state["photo"] = photo

                        canvas.configure(
                            width=disp_w,
                            height=disp_h,
                            scrollregion=(0, 0, disp_w, disp_h),
                        )
                        canvas.delete("all")
                        canvas.create_image(
                            0,
                            0,
                            anchor="nw",
                            image=photo,
                        )

                        # 阶段/提示直接覆盖在截图顶部。
                        panel_h = min(126, max(92, int(0.13 * disp_h)))
                        canvas.create_rectangle(
                            0,
                            0,
                            disp_w,
                            panel_h,
                            fill="#111827",
                            outline="",
                        )
                        canvas.create_text(
                            14,
                            12,
                            anchor="nw",
                            text=latest.stage,
                            fill="#F87171",
                            font=("Microsoft YaHei UI", 15, "bold"),
                            width=max(120, disp_w - 28),
                        )
                        canvas.create_text(
                            14,
                            46,
                            anchor="nw",
                            text=latest.hint,
                            fill="#F9FAFB",
                            font=("Microsoft YaHei UI", 11),
                            width=max(120, disp_w - 28),
                        )

                        root.geometry(f"{disp_w}x{disp_h}")

                root.after(50, process_updates)

            def on_close() -> None:
                # 用户关掉展示窗口，只关闭 UI，不停止自动求解。
                state["closed"] = True
                self.enabled = False
                try:
                    root.destroy()
                except Exception:
                    pass

            root.protocol("WM_DELETE_WINDOW", on_close)
            root.after(50, process_updates)
            root.mainloop()

        except Exception as exc:
            self._failed_reason = str(exc)
            print(f"展示窗口不可用，已自动关闭窗口功能：{exc}")


def render_marked_preview(
    frame_bgr: np.ndarray,
    marks: Sequence[ClickMark],
) -> np.ndarray:
    """测试/调试辅助：返回只画红框的截图，不打开窗口。"""
    return SolverDisplay._draw_marks(frame_bgr, marks)
