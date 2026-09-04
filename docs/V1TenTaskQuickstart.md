# V1 十任务最简测试指南

这份指南用于在同一冻结配置下人工启动 V1 的十个任务，并把每次
Codex 会话、机器人动作、checker 结果和连续仿真视频保存到
`agent_runs/`。

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
- 每个 episode 不可 resume；Codex 退出时，相应 LIBERO server
  也会退出。

## 十个任务

| 编号 | Suite | Task ID | Init state ID | Agent 收到的任务指令 |
|---:|---|---:|---:|---|
| 1 | libero_arrange_table | 0 | 0 | Arrange the table according to the provided goal image. |
| 2 | libero_arrange_table | 1 | 0 | Arrange the table. For a clean table, the butter should be placed inside the basket, and each cup should be placed on a plate. |
| 3 | robomemarena | 1 | 0 | Pick and place cookies into the basket, then pick and place tomato sauce into the same basket. |
| 4 | robomemarena | 4 | 0 | Open and close all drawers in order to check. Put butter into the drawer that already contains an object. |
| 5 | robomemarena | 7 | 0 | Pour tomato sauce over the frypan twice and place the sauce bottle into the bowl drainer. |
| 6 | robomemarena | 12 | 0 | Put cookies into the middle drawer and then put chocolate into the same drawer. |
| 7 | robomemarena | 14 | 0 | Put cookies into the top drawer and put chocolate into another drawer. |
| 8 | robomemarena | 19 | 0 | Pick and place tomato sauce, milk, and orange juice from cabinet1 to cabinet2. |
| 9 | robomemarena | 21 | 0 | Put butter into the microwave and then put chocolate into the location where the butter is placed. |
| 10 | robomemarena | 26 | 0 | Pick and place chocolate and cream from plate1 to plate2, respectively. |

V1 的 Task 4 使用 `init_state_id=0`，即隐藏物体初始位于 top drawer。
`init_state_id=1/2` 分别是已经验证过的 middle/bottom 变体，但不计入
本轮十任务结果。

## 操作者使用的 Agent 与实际被测 Agent

这里需要区分两个角色：

1. 师兄可以使用 Claude Code、Kimi、DeepSeek 或其他 coding agent
   阅读本仓库、检查实现并帮助执行下面的 shell 命令。这不会改变实验
   中实际被测的 Agent。
2. 当前 V1 launcher 实际启动并记录的是 Codex CLI harness。实验中的
   被测模型由 `--codex-model` 指定；`--codex-effort` 指定该模型支持的
   reasoning effort。

因此，`--codex-model` 可以在当前 Codex CLI 已配置且有权限访问的模型
之间切换，例如：

```bash
  --codex-model gpt-5.6-sol \
  --codex-effort high
```

省略这两个参数时，使用仓库当前默认的 `gpt-5.6-luna`、`max`。
每个实验都应显式记录这两个参数，以免把不同模型或 effort 的结果混在
一起。

Claude Code、DeepSeek 或 Kimi 本身不能通过把 `--codex-bin` 改成
`claude`、`deepseek` 或 `kimi` 来成为被测 Agent。不同 harness 的命令
行参数、MCP 注册和 session 日志格式不同；这样替换会直接破坏 launcher
契约。若要正式评测这些 Agent，需要为相应 harness 增加一个 adapter，
使它连接同一个 workspace-local MCP，并保持相同的 observation、100 次
动作预算、临时盘隔离和 `start_episode` → `osc_sequence` →
`finish_episode` 生命周期。当前 V1 尚未宣称支持该替换。

可以把下面这段话交给任意 coding agent，帮助操作者熟悉并启动当前
Codex V1，而不改变 benchmark：

```text
请阅读 docs/V1TenTaskQuickstart.md，并检查
scripts/launch_agent_episode.py、libero/libero/agent_env/ 和
libero/libero/agent_env/robomemarena_vendor/。请先解释 V1 的十个任务、
Level 4 observation、MCP 三工具接口、100 次动作预算、临时 workspace
隔离和 checker 权威性，再帮助我严格按照文档启动指定任务。不要修改
任务、checker、prompt、预算或 observation contract；如果发现设计问题，
请单独列出，不要在测试前自行修复。
```

## 从仓库根目录启动

以下命令应在 `LIBERO/` 根目录运行。把 `SUITE` 和 `TASK_ID` 换成
上表中的值；本轮固定 `SEED=100`。如果并行测试，可把不同 episode
分配到 GPU 0–3。

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
  --codex-execution-mode interactive
```

## 打开 CLI 后从哪里复制 instruction

使用 `--codex-execution-mode interactive` 时，launcher 会先创建临时
Agent workspace、启动 LIBERO server、验证 Unix socket，然后在当前
终端打开交互式 Codex CLI。它不会自动替你发送任务。

打开 CLI 前，终端会打印：

```text
run_id: <run_id>
workspace: /tmp/libero-agent-workspace-<random>
private_run: .../LIBERO/agent_runs/<run_id>
prompt_file: .../LIBERO/agent_runs/<run_id>/agent_prompt.txt

----- BEGIN TASK PROMPT -----
<完整 instruction 与机器人接口说明>
----- END TASK PROMPT -----
```

把 `BEGIN/END` 之间的完整文本作为 Codex CLI 的第一条消息粘贴进去。
也可以在另一个终端执行：

```bash
cat /绝对路径/LIBERO/agent_runs/<run_id>/agent_prompt.txt
```

复制该文件的全部内容。应当使用 launcher 打开的 Codex CLI，不要在
另一个普通 shell 中独立执行裸 `codex`：只有 launcher 启动的进程带有
本 episode 的临时 cwd、Unix socket 环境变量和 workspace-local MCP
注册。

Agent 完成任务并调用 `finish_episode` 后，等待其最终回复，然后使用
`/exit` 退出 CLI。launcher 随后收尾 server、视频和 session。若在调用
`finish_episode` 前退出 CLI，该 run 会被正确记录为 `aborted`。

## 非交互自动运行

自动正式实验只需把上面的参数改为：

```bash
  --codex-execution-mode exec
```

这是默认模式。launcher 会把同一个 `agent_prompt.txt` 直接作为
`codex exec` 的首条消息，其他任务、接口、预算和隔离语义完全相同。

## 结果与回看

每次结果位于：

```text
agent_runs/<run_id>/
  agent_prompt.txt
  run_manifest.json
  actions.jsonl
  result.json
  continuous_video.mp4
  codex_session.jsonl
  private_observations/
```

`result.json` 是最终 checker 结果；`continuous_video.mp4` 是连续仿真
录像；`codex_session.jsonl` 和 `actions.jsonl` 用于审计 Agent 的观察、
分析和机器人调用。

启动只读 Viewer：

```bash
PYTHONPATH=. ../miniconda3/envs/libero/bin/python \
  scripts/run_agent_viewer.py \
  --host 0.0.0.0 \
  --port 8765 \
  --runs-root agent_runs
```

打开命令打印的浏览器地址，即可按 run 查看 Agent timeline、历史观测、
动作和连续视频。
