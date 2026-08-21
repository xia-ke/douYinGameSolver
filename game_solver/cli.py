from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adb import ensure_adb
from .config import DEFAULT_SLOTS
from .engine import analyze_image, run_auto_flow_mode, run_manual_step_mode

def main() -> int:
    parser = argparse.ArgumentParser(
        description="游戏截图滚动求解器 v5.2：动态队列 + 第6停车位修复 + 纯数字像素监控"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--adb", action="store_true", help="使用 ADB")
    src.add_argument("--image", type=Path, help="分析一张本地截图")

    parser.add_argument("--serial", help="ADB 设备序列号（多设备时使用）")
    parser.add_argument("--state", type=Path, default=Path("solver_state.npz"), help="持久化状态文件")
    parser.add_argument("--shots-dir", type=Path, default=Path("solver_shots"), help="ADB 截图保存目录")
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS, help="停车位总数；按本游戏规则默认固定为 6")
    parser.add_argument("--reset", action="store_true", help="新局重建动态颜色、棋盘与空停车区参考")
    parser.add_argument("--no-auto-tap", action="store_true", help="不自动点击；自动模式下一次分析后退出")
    parser.add_argument("--manual-step", action="store_true", help="人工逐步调试模式")
    parser.add_argument("--tap-delay", type=float, default=0.35, help="分析结束到点击之间等待秒数，默认 0.35")
    parser.add_argument("--double-step-gap", type=float, default=0.35, help="停车空位>=3时连续两步之间的点击间隔，默认 0.35 秒")
    parser.add_argument("--flow-start-delay", type=float, default=2.5, help="点击后多久建立停车数字监控基准，默认 2.5 秒")
    parser.add_argument("--parking-check-interval", type=float, default=1.0, help="停车数字区域检查间隔，默认 1 秒")
    parser.add_argument("--parking-idle-timeout", type=float, default=8.0, help="停车数字连续无变化多久算分流结束，默认 20 秒")
    parser.add_argument("--monitor-max-failures", type=int, default=5, help="停车数字监控连续失败多少次后安全停止，默认 5")
    parser.add_argument("--empty-settle-delay", type=float, default=0.5, help="停车数字单帧为空后的连续确认间隔，默认 0.5 秒")
    parser.add_argument("--queue-empty-confirm-delay", type=float, default=0.5, help="排队区判空后的第二次确认间隔，默认 0.5 秒")
    parser.add_argument("--analysis-settle-delay", type=float, default=0.35, help="停车监控结束到下一轮重新截图前的收尾等待，默认 0.35 秒")
    parser.add_argument("--skip-sixth-slot-unlock", action="store_true", help="新局时跳过广告解锁第6停车位")
    parser.add_argument("--unlock-ad-wait", type=float, default=15.5, help="点击解锁后至少等待多久再关闭广告，默认 15.5 秒")
    parser.add_argument("--unlock-return-settle-delay", type=float, default=1.0, help="广告关闭返回游戏后再等待多久开始分析，默认 1 秒")
    # 兼容旧版本命令行；纯数字监控模式下该参数不再使用。
    parser.add_argument("--transition-timeout", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.slots < 1:
        parser.error("--slots 必须 >= 1")
    for name in ("tap_delay", "double_step_gap", "flow_start_delay", "empty_settle_delay", "queue_empty_confirm_delay", "analysis_settle_delay", "unlock_ad_wait", "unlock_return_settle_delay"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} 必须 >= 0")
    if args.parking_check_interval <= 0:
        parser.error("--parking-check-interval 必须 > 0")
    if args.parking_idle_timeout <= 0:
        parser.error("--parking-idle-timeout 必须 > 0")
    if args.monitor_max_failures < 1:
        parser.error("--monitor-max-failures 必须 >= 1")

    try:
        if args.image:
            result = analyze_image(args.image, args.state, args.reset, args.slots)
            print(result.report)
            return 0

        ensure_adb(args.serial)
        args.shots_dir.mkdir(parents=True, exist_ok=True)
        if args.manual_step:
            return run_manual_step_mode(args)
        return run_auto_flow_mode(args)

    except KeyboardInterrupt:
        print("\n用户中断，已停止自动流程。")
        return 130
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        return 1