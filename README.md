# game_solver v5.1 dynamic queue

运行：

```powershell
python game_solver_v5.py --adb --reset
```

同一局后续再次启动时不要加 `--reset`：

```powershell
python game_solver_v5.py --adb
```

模块：

- `game_solver/config.py`：常量与阈值
- `game_solver/models.py`：数据结构
- `game_solver/board.py`：棋盘、调色板、可达区域
- `game_solver/ocr.py`：快速 OCR / 慢速兜底
- `game_solver/vehicles.py`：排队车辆与停车数字锚点
- `game_solver/strategy.py`：硬安全规则与候选评分
- `game_solver/state.py`：状态持久化
- `game_solver/adb.py`：ADB 截图与点击
- `game_solver/monitor.py`：纯停车数字像素变化监控（不 import OCR）
- `game_solver/engine.py`：单轮分析与自动循环
- `game_solver/cli.py`：命令行参数

v5 的分流监控与 OCR 完全解耦。`monitor.py` 只看固定停车数字窄带里的白色数字像素变化。


## v5.1 变化

- 第一排车辆列数与横坐标不再固定为 4 列。
- 程序从第一排白色数字自动检测当前列中心；已验证 4 列和 5 列截图。
- 点击使用本轮识别到的车辆真实坐标，不再按固定列坐标反推。
- 排队区判空也使用动态列检测，避免 5 列关卡被误判为胜利。


### 已验证的动态队列布局

- 旧关卡：4 列，中心约 `240, 393, 547, 700`
- 本次关卡：5 列，中心约 `163, 316.5, 469.5, 623.5, 777`

程序不再使用固定 `FRONT_X_N` 进行分析、判空或点击。
