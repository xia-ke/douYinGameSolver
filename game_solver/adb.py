from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def adb_prefix(serial: Optional[str]) -> List[str]:
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    return cmd


def ensure_adb(serial: Optional[str]) -> None:
    if shutil.which("adb") is None:
        raise RuntimeError("找不到 adb。请安装 Android platform-tools，并把 adb 加入 PATH。")
    p = subprocess.run(adb_prefix(serial) + ["get-state"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0 or "device" not in p.stdout:
        raise RuntimeError("ADB 未连接到可用设备。请确认 USB 调试已开启并执行 adb devices 检查。\n" + p.stderr.strip())


def adb_capture_png_bytes(serial: Optional[str]) -> bytes:
    # 启动时已经检查过 ADB；监控阶段不再每秒额外 get-state。
    p = subprocess.run(adb_prefix(serial) + ["exec-out", "screencap", "-p"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError("ADB 截图失败: " + p.stderr.decode(errors="ignore"))
    return p.stdout


def adb_capture_bgr(serial: Optional[str]) -> np.ndarray:
    raw = adb_capture_png_bytes(serial)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("ADB 截图解码失败。")
    return img


def adb_screencap(out_path: Path, serial: Optional[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(adb_capture_png_bytes(serial))


def save_bgr(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise RuntimeError(f"无法保存截图: {path}")


def adb_tap(x: int, y: int, serial: Optional[str]) -> None:
    p = subprocess.run(
        adb_prefix(serial) + ["shell", "input", "tap", str(int(x)), str(int(y))],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if p.returncode != 0:
        raise RuntimeError("ADB 点击失败: " + p.stderr.strip())


def shot_stamp() -> str:
    now = time.time()
    base = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    ms = int((now - int(now)) * 1000)
    return f"{base}_{ms:03d}"
