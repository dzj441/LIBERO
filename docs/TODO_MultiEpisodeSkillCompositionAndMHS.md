结论：这个“同一 context 连做两个子任务，再到新场景做组合任务”的实验非常可行，而且 LIBERO 已有现成任务，不必先手工拆 BDDL。它比现在的静态 ICL 更接近“Agent 从自己的具身经历中组合技能”。

> 实现状态（2026-08-29）：MultiEpisodeService、MCP 生命周期、逐 episode
> P4 ICL 切换、独立审计目录/视频以及 Viewer 分段均已实现。真实 Codex
> curriculum、独立 E3 baseline 与独立 E3 fixed-demo 对照均已完成首轮运行。

### 1. 现成任务组合

推荐按一个 Codex session 连续执行：

1. 开抽屉
   libero_90 task 7：KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet
   libero/libero/bddl_files/libero_90/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet.bddl:3

2. 把碗放进已经打开的抽屉
   libero_90 task 29：KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet
   其 init 明确包含抽屉已打开：libero/libero/bddl_files/libero_90/KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.bddl:79

3. 换到新场景完成组合任务
   libero_goal task 3：open_the_top_drawer_and_put_the_bowl_inside
   libero/libero/bddl_files/libero_goal/open_the_top_drawer_and_put_the_bowl_inside.bddl:3

三者的 init 文件和专家 HDF5 均已存在。MultiEpisodeService 支持每段独立选择
`icl=none|fixed_demo`。当前主要 curriculum 条件为 E1/E2 提供各自的 P4 fixed
demo、E3 不提供 demo，用于比较“新 context 直接做组合任务”和“同一 context
先获得两段在线具身经历”之间的差异；三段均有 fixed demo 的配置仍保留为直接
ICL 对照。

最重要的对照是：

- Baseline：新 context 直接做第 3 个组合任务。
- Curriculum：同一 context 做 1 → 2 → 3。
- P4、native OSC、模型与预算保持一致。
- 最终比较第 3 个任务成功率。

首次可以只跑一个 pilot；要谈成功率，至少再换 3 个目标 seed。前两个 episode 是否成功也要分别记录，因为“看过失败经验后学会”与“成功掌握子技能后迁移”意义不同。

### 2. 已实现架构

现有单任务 `AgentEpisodeService` 保持不变，外层 `MultiEpisodeService` 负责：

- 只启动一个 Codex 进程和一个 session。
- 每个 episode 仍使用 start → osc-sequence → finish。
- finish 后关闭当前 LIBERO 环境并准备下一场景。
- 下一次 start 返回新任务 instruction 和首帧。
- 每个 episode 单独保存 checker、actions、observation 和视频。
- session JSONL 只有一份，Viewer 按 episode 分段展示。
- 最后一个 episode finish 后才结束 Codex/server。

当前实现额外保证：

- 未来任务 instruction 不写入初始 Agent prompt，只由对应 episode 的
  `start_episode` 返回；
- `finish_episode` 明确返回 `next_episode_available` 和
  `curriculum_complete`；
- `benchmark_inputs/current_observation/` 与 `expert_demo/` 都在切换时原子
  更新，workspace 中不会同时主动提供多个 episode 的 ICL；
- 根 `actions.jsonl` 带 `episode_index`，每个 `episodes/episode_NNN/` 仍独立
  保存 actions、result、private observations 和 continuous video；
- 根结果的 `success`/`final_episode_success` 表示组合目标 episode，另以
  `all_episodes_success` 记录三段是否全部成功。

默认实验计划位于：

```text
configs/agent_curricula/drawer_skill_composition_p4_fixed_demo.json
```

启动命令：

```bash
PYTHONPATH=. ../miniconda3/envs/libero/bin/python \
  scripts/launch_agent_curriculum.py \
  --curriculum-plan \
    configs/agent_curricula/drawer_skill_composition_p4_fixed_demo.json \
  --codex-model gpt-5.6-sol \
  --codex-effort high
```

控制接口与 observation schema都不用改。这属于中等开发量，不涉及重写 LIBERO backend。

### 3. “物理 MCP”到底是什么

文章标题不太准确。Anthropic 新发布的是 MHS（Model Hardware Standard），并不是 MCP 的新版本。

MHS 是硬件驱动规范；MCP、CLI 和 API 是调用 MHS 的三种方式。它目前仍是有限研究预览，规范尚未开源，因此现在无法声称严格兼容 MHS。Anthropic 官方公告

这反而说明我们现在的设计没有走错：

- AgentEpisodeService 类似设备 driver。
- .libero/episode.json 类似能力与限制 manifest。
- liberoctl 正好对应 MHS 明确支持的 CLI 访问方式。
- osc-sequence 对应它提到的将快速、长时间操作批量提交给设备执行。

### 4. MCP 状态

当前默认控制链路已经是 MCP + native OSC sequence，CLI adapter 仍保留为显式
兼容选项。curriculum 复用同一 MCP server 和三项工具，不增加 task-specific tool：

- `start_episode`
- `osc_sequence`
- `finish_episode`

这仍不是对未公开 MHS 规范的兼容性声明。等 MHS 正式开源后，再单独映射其
states、procedures、telemetry 和 safety tags；不要把该变化混入 curriculum
能力实验。

另外，Anthropic 自己最新的机器人实验也采用 LIBERO、7D EEF 命令、视觉和力反馈，并强调控制接口会显著影响测得的 Agent 能力；这与我们当前方向高度一致。Anthropic LIBERO 机器人实验

所以当前最高价值的下一步是重复 multi-episode same-context 实验，并正交比较
动作预算与是否显式提示经验迁移；更深的 MHS 对齐优先级在其后。
