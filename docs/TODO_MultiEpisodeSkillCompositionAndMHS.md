结论：这个“同一 context 连做两个子任务，再到新场景做组合任务”的实验非常可行，而且 LIBERO 已有现成任务，不必先手工拆 BDDL。它比现在的静态 ICL 更接近“Agent 从自己的具身经历中组合技能”。

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

三者的 init 文件也都已存在。这个实验暂时不需要专家 HDF5，因为前两个 episode 本身就是 Agent 在同一 session 中获得的在线经验。

最重要的对照是：

- Baseline：新 context 直接做第 3 个组合任务。
- Curriculum：同一 context 做 1 → 2 → 3。
- P4、native OSC、模型与预算保持一致。
- 最终比较第 3 个任务成功率。

首次可以只跑一个 pilot；要谈成功率，至少再换 3 个目标 seed。前两个 episode 是否成功也要分别记录，因为“看过失败经验后学会”与“成功掌握子技能后迁移”意义不同。

### 2. 当前架构需要的改动

目前 libero/libero/agent_env/service.py:1 明确定义为 one-episode service，finish 后整个运行结束。

需要增加一个外层 MultiEpisodeService：

- 只启动一个 Codex 进程和一个 session。
- 每个 episode 仍使用 start → osc-sequence → finish。
- finish 后关闭当前 LIBERO 环境并准备下一场景。
- 下一次 start 返回新任务 instruction 和首帧。
- 每个 episode 单独保存 checker、actions、observation 和视频。
- session JSONL 只有一份，Viewer 按 episode 分段展示。
- 最后一个 episode finish 后才结束 Codex/server。

控制接口与 observation schema都不用改。这属于中等开发量，不涉及重写 LIBERO backend。

### 3. “物理 MCP”到底是什么

文章标题不太准确。Anthropic 新发布的是 MHS（Model Hardware Standard），并不是 MCP 的新版本。

MHS 是硬件驱动规范；MCP、CLI 和 API 是调用 MHS 的三种方式。它目前仍是有限研究预览，规范尚未开源，因此现在无法声称严格兼容 MHS。Anthropic 官方公告

这反而说明我们现在的设计没有走错：

- AgentEpisodeService 类似设备 driver。
- .libero/episode.json 类似能力与限制 manifest。
- liberoctl 正好对应 MHS 明确支持的 CLI 访问方式。
- osc-sequence 对应它提到的将快速、长时间操作批量提交给设备执行。

### 4. 是否需要改成 MCP

技术上很容易，但不建议现在替换 CLI。最干净的结构是以后增加并行适配层：

2. 将结果作为稳定 baseline。
3. 再在独立分支增加 MCP adapter。
4. 用同一任务比较 CLI 与 MCP，而不是把接口变化和 curriculum 同时引入。
5. 等 MHS 正式开源后，再映射其 states、procedures、telemetry 和 safety tags。

另外，Anthropic 自己最新的机器人实验也采用 LIBERO、7D EEF 命令、视觉和力反馈，并强调控制接口会显著影响测得的 Agent 能力；这与我们当前方向高度一致。Anthropic LIBERO 机器人实验

所以当前最高价值的下一步是多 episode 同 context 实验；MCP/MHS 对齐可行，但优先级应放在该实验之后。
