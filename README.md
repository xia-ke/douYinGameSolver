# douYinGameSolver

一个面向当前抖音小游戏关卡的 **mission-first 自动求解器**。目标不是维护历史修补链，而是让每一步自动点击只建立在当前稳定截图、明确的观测可信度和已确认游戏规则上。

核心闭环：

```text
stable frame
→ current-frame observation
→ validation / ObservationHealth
→ trusted-only planning
→ deterministic flow closure + hard stable safety
→ lexicographic utility ranking
→ ADB execute
→ monitor stable
→ repeat
```

## 安装

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

离线回归测试额外安装：

```powershell
python -m pip install -r requirements-test.txt
```

运行自动求解需要本机可调用 `adb`，并且目标 Android 设备已授权连接。

## 运行

新局首次启动：

```powershell
python game_solver_v5.py --adb --reset
```

同一局继续运行时复用 `solver_state.npz`，不要重复 `--reset`：

```powershell
python game_solver_v5.py --adb
```

只分析一张本地稳定截图、不执行 ADB 点击：

```powershell
python game_solver_v5.py --image .\solver_shots\analysis_xxx.png --reset
```

常用诊断模式：

```powershell
python game_solver_v5.py --adb --no-auto-tap
python game_solver_v5.py --adb --manual-step
```

默认会在新局检查第 6 个停车位的解锁流程；如果已确认不需要该流程，可显式使用 `--skip-sixth-slot-unlock`。

## 严格观测门

每张稳定截图都会重新构造当前 52×38 棋盘，单元格状态为 `COLOR / EMPTY / UNKNOWN`。**当前稳定帧是空间状态的唯一权威来源**；上一份 trusted state 只是历史上下文，只能用于解析当前 `UNKNOWN` 或校验不变量，不能把旧棋盘直接复制成当前棋盘。

`ObservationHealth` 汇总当前帧的棋盘转移、OCR、停车位数量和容量守恒等校验结果：

- `trusted=true`：允许进入正常规划；
- `trusted=false`：自动模式先做 bounded retry，重试期间不提交状态、不点击；
- bounded retry 后仍不可信：默认 `NO_CLICK_UNTRUSTED`，保留上一份 trusted state 和上一动作 checkpoint，继续重新观测。

`--experimental-continue` 只是显式诊断开关，不是默认求解策略。启用后，不可信 observation 在 bounded retry 后可以提交当前保守状态，但只生成**单步候选**，不会生成两步计划。

容量侧信道只做**数量校验**，不会根据容量差额猜测具体哪个棋盘坐标消失。

## 已确认的游戏规则与边界

### 停车位硬安全

停车位总数默认是 6。已确认规则是：**分流稳定后停车仍为 6/6 即失败**。因此策略先做 hard feasibility；任何预测稳定占用上界达到总停车位的候选都会直接拒绝，后续 utility 不能把它重新“加分救活”。

### 从下方可达

棋盘 reachability 只从棋盘下方的开放区域传播。被下方色块阻隔的小色块不会凭空消失；只有在连接到开放区域后才可能进入可吸收集合。`UNKNOWN` 作为保守阻挡处理，不用于凭空开路。

### 同色车辆的低 remain 优先

对同色停车车辆，已确认**剩余数字更小的车优先获得供给**。一旦开始吸收，remain 继续下降，因此会持续保持优先级直到完成。该规则参与确定性完成数与停车释放证明。

### nearest-cell 的置信边界

“车辆优先吸收最近的可移动同色色块”目前只作为弱空间前瞻，不属于 hard safety 事实：

- 只对已经在停车位、位置已知的车辆使用；
- 本轮新点击的同色车停车位置未知时不猜；
- 同 remain 分配平局不猜；
- 最近两个候选距离过近时主动 abstain；
- nearest 结果只能进入 utility 最后的 bounded tie-break，不能修改 `stable_safe` 或确定性完成证明。

对应历史截图尚未保留，因此 nearest 的真实像素 replay 仍是 `pending_fixture`。

## 策略

`strategy.py` 使用确定性的 `simulate_flow_closure()` 推演本轮稳定结果。处理顺序是：

1. 先过滤不满足稳定停车 hard safety 的动作；
2. 安全候选按词典序 utility 比较：
   - 已有停车车保证释放数；
   - 所有车辆保证完成数；
   - 确定性清除色块数；
   - 对当前 front / parked / next-row 有用颜色的确定性新暴露；
   - 队列推进；
   - 最后的有界启发式 tie-break。
3. 两步计划使用同一套规则 utility，并且第一步单独执行也必须安全；不允许靠 tie-break 把本来没有规则级增益的两步计划抬到单步之上。

## 状态文件

当前 state schema 只保存上一份 **trusted observation context**：palette、上一份 trusted grid、turn、屏幕尺寸、空停车区参考和可选的 grid RGB snapshot。

- 当前 schema 的 `STATE_VERSION = 4`；
- 不兼容的旧 state 不在程序内部做双读/迁移；请在新局执行一次 `--reset` 重建；
- 正常严格模式下，只有 trusted observation 才能成为后续历史上下文；
- retry observation 是只读的，不会覆盖已提交 state。

## Replay regression

离线回归入口：

```powershell
pytest tests\replay\
```

`tests/replay/cases.json` 明确区分：

- `active`：真实来源 artifact 已保留，并且存在可执行 runner；
- `pending_fixture`：历史行为和预期已经登记，但真实截图/日志缺失，不能被伪装成已通过的像素证据。

当前登记的历史 C01/C08、OCR 26→20、三列界面、nearest-cell 和同色低 remain 图像/转移案例都因为原始真实截图缺失而保持 `pending_fixture`。其中已确认的纯规则可以由独立 deterministic test 保护，但不能替代缺失的真实截图 replay。

未来要把 `pending_fixture` 提升为 `active`，必须来自自然发生并保留下来的真实稳定截图/日志，补齐 artifact 路径和 subsystem runner，然后先证明 replay 会对错误预期失败，再修改生产逻辑。不要重建或合成历史截图来“补通过”。详细流程见 `tests/replay/README.md`。

## 诊断输出

默认 ADB 自动模式会在 `solver_shots/` 保存稳定截图，并维护：

- `decision_log.txt`：当前 observation、候选 utility、最终计划和执行结果；
- `color_log.txt`：palette、棋盘颜色矩阵、队列/停车颜色；
- `number_log.txt`：第一排、第二排和停车数字。

这些展示/日志格式集中在 `game_solver/debug.py`，不会重新执行 OCR 或改变策略状态。

## 当前模块职责

- `game_solver/config.py`：当前固定版式、检测阈值与 state schema 常量。
- `game_solver/models.py`：`Car`、`Candidate`、`ObservationHealth`、`AnalysisResult` 等数据结构。
- `game_solver/board.py`：current-frame 棋盘观测、palette、bottom-only reachability 与观测校验。
- `game_solver/ocr.py`：排队区、停车区共用的游戏数字识别器与结构化 OCR diagnostics。
- `game_solver/vehicles.py`：第一/第二排车辆、停车车辆与颜色/数字提取。
- `game_solver/strategy.py`：确定性 flow closure、hard feasibility、lexicographic utility 和两步规划。
- `game_solver/state.py`：唯一的 previous-trusted-context state schema。
- `game_solver/adb.py`：ADB 截图与点击。
- `game_solver/monitor.py`：停车数字像素变化监控；与 OCR 解耦。
- `game_solver/unlock.py`：新局第 6 停车位解锁流程与游戏界面判定。
- `game_solver/debug.py`：report、decision/color/number logs 与 observation diagnostics。
- `game_solver/engine.py`：capture → perception → trust → plan → trusted-context commit → execute → monitor 的运行编排。
- `game_solver/cli.py`：命令行入口与参数校验。

## 当前验证边界

`pytest tests/replay/` 可以离线验证当前 deterministic contracts，并持续显示缺少真实 artifact 的 `pending_fixture` 案例。仓库当前没有保存可用于历史像素回归的真实截图，因此真实设备行为和本地 `--image` 像素路径仍应在新获得的真实稳定截图上验证；不要把缺失 fixture 当成通过。
