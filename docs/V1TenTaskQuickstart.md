# V1 十任务最简测试指南

这份指南用于在同一冻结配置下启动 V1 的十个任务。仿真与 Agent
分开启动：LIBERO 提供相同的 MCP、observation 和 checker，不限定使用
Codex、Claude Code 或其他支持 STDIO MCP 的 Agent。机器人动作、checker
结果和连续仿真视频统一保存在 `agent_runs/`。

## 统一配置

- Observation profile：Level 4。
- ICL：关闭。
- 动作：LIBERO 原生 normalized `OSC_POSE`，每次向 `osc_sequence`
  提交 1–20 个 7D micro-actions。
- Transport：workspace-local MCP，只注册 `start_episode`、
  `osc_sequence`、`finish_episode`。
- 动作预算：每个 episode 最多 100 次 `osc_sequence` 调用。
- Workspace：默认使用系统临时盘中的随机目录；不要添加
  `--keep-workspace`。episode 结束后由操作系统负责清理。
- 仿真：EGL；launcher 自动选择与当前 NVIDIA 驱动匹配的
  `runtime/nvidia/` userspace libraries。
- 每个 episode 不可 resume；调用 `finish_episode` 后 LIBERO server
  自动退出，未完成时可在仿真终端按 Ctrl-C 中止。

## 十个任务

| 编号 | Suite | Task ID | Init state ID | Agent 收到的任务指令 | 预置完整 prompt |
|---:|---|---:|---:|---|---|
| 1 | libero_arrange_table | 0 | 0 | Arrange the table according to the provided goal image. | `benchmark_configs/v1_10task/prompts/01_arrange_table_goal_image.txt` |
| 2 | libero_arrange_table | 1 | 0 | Arrange the table. For a clean table, the butter should be placed inside the basket, and each cup should be placed on a plate. | `benchmark_configs/v1_10task/prompts/02_arrange_table_text.txt` |
| 3 | robomemarena | 1 | 0 | Pick and place cookies into the basket, then pick and place tomato sauce into the same basket. | `benchmark_configs/v1_10task/prompts/03_robomemarena_task_01.txt` |
| 4 | robomemarena | 4 | 0 | Open and close all drawers in order to check. Put butter into the drawer that already contains an object. | `benchmark_configs/v1_10task/prompts/04_robomemarena_task_04.txt` |
| 5 | robomemarena | 7 | 0 | Pour tomato sauce over the frypan twice and place the sauce bottle into the bowl drainer. | `benchmark_configs/v1_10task/prompts/05_robomemarena_task_07.txt` |
| 6 | robomemarena | 12 | 0 | Put cookies into the middle drawer and then put chocolate into the same drawer. | `benchmark_configs/v1_10task/prompts/06_robomemarena_task_12.txt` |
| 7 | robomemarena | 14 | 0 | Put cookies into the top drawer and put chocolate into another drawer. | `benchmark_configs/v1_10task/prompts/07_robomemarena_task_14.txt` |
| 8 | robomemarena | 19 | 0 | Pick and place tomato sauce, milk, and orange juice from cabinet1 to cabinet2. | `benchmark_configs/v1_10task/prompts/08_robomemarena_task_19.txt` |
| 9 | robomemarena | 21 | 0 | Put butter into the microwave and then put chocolate into the location where the butter is placed. | `benchmark_configs/v1_10task/prompts/09_robomemarena_task_21.txt` |
| 10 | robomemarena | 26 | 0 | Pick and place chocolate and cream from plate1 to plate2, respectively. | `benchmark_configs/v1_10task/prompts/10_robomemarena_task_26.txt` |

V1 的 Task 4 使用 `init_state_id=0`，即隐藏物体初始位于 top drawer。
`init_state_id=1/2` 分别是已经验证过的 middle/bottom 变体，但不计入
本轮十任务结果。

## 第一步：启动仿真

以下命令应在 `LIBERO/` 根目录运行。把 `SUITE` 和 `TASK_ID` 换成
上表中的值；本轮固定 `SEED=100`。如果并行测试，可把不同 episode
分配到 GPU 0–3。这个终端在整个 episode 中需要保持运行。

```bash
SUITE=libero_arrange_table
TASK_ID=0
SEED=100
GPU=0

PYTHONPATH=. ../miniconda3/envs/libero/bin/python \
  scripts/launch_agent_episode.py \
  --suite "$SUITE" \
  --task-id "$TASK_ID" \
  --init-state-id 0 \
  --profile level4 \
  --seed "$SEED" \
  --render-gpu-device-id "$GPU" \
  --max-agent-steps 100 \
  --action-interface native_osc_sequence \
  --control-transport mcp \
  --icl none \
  --external-agent
```

仿真 ready 后会打印四个路径：

```text
run_id: <run_id>
workspace: /tmp/libero-agent-workspace-<random>
private_run: .../LIBERO/agent_runs/<run_id>
prompt_file: .../LIBERO/agent_runs/<run_id>/agent_prompt.txt
mcp_config_file: .../LIBERO/agent_runs/<run_id>/agent_mcp_config.json
```

其中：

- `workspace` 是 Agent 必须进入的临时工作目录；本次公开的
  `current_observation/` 会原子更新在这里。
- `agent_mcp_config.json` 是本 episode 的标准 STDIO MCP 配置，只注册
  `start_episode`、`osc_sequence` 和 `finish_episode`。
- `agent_prompt.txt` 是本次实际使用的完整 prompt，与上表对应的仓库
  预置 TXT 一致。运行目录中的副本用于审计。

## 第二步：让所选 Agent 给出自己的接入方法

Claude Code、Codex、Kimi、DeepSeek 等 Agent 的 MCP 注册参数和
非交互命令可能不同，因此本文不维护某一个 Agent 的专用命令。正式测试
前，在仓库根目录另开一个仅用于接入准备的 setup session，把下面这段
话交给准备使用的 Agent：

```text
请作为 LIBERO benchmark 的接入助手，阅读
docs/V1TenTaskQuickstart.md、scripts/launch_agent_episode.py 和
scripts/libero_mcp_server.py；如有必要，可继续查看与 Agent 接入直接相关的
代码。请结合你当前 Agent/CLI 的实际名称、版本和本机 help，告诉操作者：

1. 如何在指定 WORKSPACE 中启动一个全新 session；
2. 如何为该 session 加载 MCP_CONFIG 指向的 STDIO MCP；
3. 如何像 codex exec 一样，把 PROMPT_FILE 作为首条消息非交互执行；
4. 如何进入交互 CLI，再由操作者粘贴 PROMPT_FILE；
5. 如何保存该 Agent 自己的 session 日志用于审计。

launcher 会提供 WORKSPACE、MCP_CONFIG 和 PROMPT_FILE 的绝对路径。
MCP_CONFIG 使用顶层 mcpServers 格式，其中 libero 条目已经包含 command、
args 和 env。如果你的客户端格式不同，只映射这三个字段，不修改 MCP
server、任务 prompt、动作预算、observation 或 checker。

请只返回经过当前 CLI help 核验的具体命令、必要适配和注意事项；不要启动
仿真或 rollout，不要修改代码，也不要查看任务 BDDL、checker、私有
observation 或已有实验结果。
```

这一步可以由师兄惯用的任何 coding agent 完成。它的职责是研究“自己
如何挂载这个 MCP”，不是参加具身评测。

为了避免 codebase、checker 或历史结果进入被测上下文，setup session
不能直接续作正式 rollout。拿到接入命令后应关闭它，再从 launcher 打印的
临时 `workspace` 启动一个全新的被测 session。

## 第三步：启动正式 Agent

在第二个终端记录仿真 launcher 打印的三个路径：

```bash
WORKSPACE=/tmp/libero-agent-workspace-<random>
MCP_CONFIG=/绝对路径/LIBERO/agent_runs/<run_id>/agent_mcp_config.json
PROMPT_FILE=/绝对路径/LIBERO/agent_runs/<run_id>/agent_prompt.txt
```

然后严格使用 setup Agent 给出的本机命令：

1. 从 `WORKSPACE` 启动全新 Agent session；
2. 只为该 session 加载 `MCP_CONFIG`；
3. 二选一：以非交互模式直接发送 `PROMPT_FILE`，或打开交互 CLI 后粘贴
   该文件的完整内容；
4. 不向正式 session 提供 LIBERO 仓库、checker 或历史 run 作为上下文。

若所选 Agent 原生支持 STDIO MCP，通常只需改变其 MCP 加载参数和启动
命令，不需要修改本仓库代码。若它不支持此格式，再根据 setup Agent 的
结论增加独立的薄 adapter。

## 第四步：完成与收尾

Agent 必须按 prompt 使用 `start_episode` → `osc_sequence` →
`finish_episode`。`finish_episode` 返回官方 checker 结果后，仿真终端会
自动退出并写完视频。若 Agent 在 finish 前退出，在仿真终端按 Ctrl-C；
该 run 会记录为 `aborted`，不能 resume 当前物理 episode。

## 可选：仍由 launcher 直接启动 Codex

已有 Codex 一键模式继续保留。去掉 `--external-agent`，并选择：

```bash
--codex-execution-mode exec         # 自动执行
--codex-execution-mode interactive  # 打开 CLI 后手工粘贴 prompt
```

这两种模式由 launcher 自动注入 Codex MCP。使用 Claude Code、Kimi、
DeepSeek 等其他 Agent 时，使用上面的 external workflow。

## 结果与回看

每次结果位于：

```text
agent_runs/<run_id>/
  agent_prompt.txt
  agent_mcp_config.json
  run_manifest.json
  actions.jsonl
  result.json
  continuous_video.mp4
  codex_session.jsonl       # 仅 launcher 直接启动 Codex 时存在
  private_observations/
```

`result.json` 是最终 checker 结果；`continuous_video.mp4` 是连续仿真
录像；`actions.jsonl` 记录机器人调用。`codex_session.jsonl` 只由内置
Codex harness 自动归档；外部 Agent 的会话日志由相应 Agent 自己保存。

启动只读 Viewer：

```bash
PYTHONPATH=. ../miniconda3/envs/libero/bin/python \
  scripts/run_agent_viewer.py \
  --host 0.0.0.0 \
  --port 8765 \
  --runs-root agent_runs
```

打开命令打印的浏览器地址，即可按 run 查看历史观测、动作和连续视频；
只有归档了受支持 session JSONL 的 run 才会额外显示完整 Agent timeline。
