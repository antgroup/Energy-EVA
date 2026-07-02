# 电力交易标准评测集治理说明

`power_trade_standard_v1` 是 AgentX Bench 电力交易客户侧标准评测集，不再混用历史 QA/v1.4 语料。

## 基准信息

- 默认评测日期：2026-06-23

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `suite.yaml` | 数据集元信息、覆盖度、caseType 和规则 |
| `cases.jsonl` | 43 条 active benchmark case |
| `scoring.yaml` | 面向 Agent 能力的评分配置 |
| `source_row_audit.jsonl` | 源表非空行处理审计清单 |
| `curation_guide.md` | 本治理说明 |

## 最小 Case 结构

每条 active case 只依赖当前 AgentX Bench 和 evaluation service 必需字段：

```json
{
  "caseId": "power_trade_standard_v1_001",
  "name": "可读标题",
  "inputJson": {
    "query": "Agent 实际可见的问题，必须包含关键上下文",
    "teamName": "${subject.teamName}"
  },
  "expectedJson": {
    "intent": "canonical.intent",
    "expectedAnswer": "核心结论、关键事实和必要业务边界",
    "answerCriteria": ["评分拆解点"]
  },
  "attributesJson": {
    "category": "category_name",
    "caseType": "business_answer",
    "difficulty": "L2"
  }
}
```

`expectedAnswer` 是自动评分的核心参考，必须承载正确答案本身。`answerCriteria` 只拆解评分点、完整性、风险边界和可执行性，不能成为唯一正确答案来源。

## 源表处理规则

| 原始内容类型 | 处理方式 |
| --- | --- |
| 明确用户问题 | 转为 active case |
| 重复问题 | 合并到同一 active case，并在审计文件中标记 `merged` |
| 产品设计说明 | 不原样入库，转化为具体 case 或治理规则，并标记 `transformed` |
| 空行 | 不进入审计和 active case |
| 期望不完整 | 补齐 `expectedAnswer` 和 `answerCriteria` 后入库 |
| 联系人、手机号、客户敏感信息 | 删除或替换为泛化主体 |

`source_row_audit.jsonl` 为每个非空源表行保留一条记录，字段包括 `sourceRowId`、`sanitizedQuestion`、`sourceIntent`、`decision`、`caseIds`、`reason` 和 `piiHandling`。

## Category

| Category | 说明 |
| --- | --- |
| `retail_package` | 售电套餐、账单分析、用户侧套餐推荐 |
| `trade_brief` | 盘前交易简报 |
| `market_query` | 出清价、价差、竞价空间等市场问数 |
| `policy_rule` | 政策、规则、入市、合规、价格机制问答 |
| `external_signal` | 新闻、政策传言、社交情绪、突发事件影响分析 |
| `intraday_strategy` | 盘中滚撮、盘口行为、成交进度和交易策略 |
| `spot_strategy` | 日前现货申报策略 |
| `forecast_analysis` | 电价、负荷、出力、气象、能量块预测 |
| `trade_review` | 现货复盘、价格波动原因分析 |
| `system_data_query` | 主体申报、开停机、历史实际电量等系统数据查询 |

## CaseType

| CaseType | 评分侧重点 |
| --- | --- |
| `data_query` | 数据查询、口径、时间范围、单位和结果解释 |
| `business_answer` | 业务回答完整性、事实边界、政策/套餐/复盘口径 |
| `scenario_analysis` | 给定外部信号或场景下的分析路径、方向性和不确定性 |
| `strategy_advice` | 交易动作、量价边界、触发条件和风险控制 |
| `robustness` | 缺信息、能力边界和不确定性下的保守处理 |

## 盘中滚撮口径

不做连续仿真，不要求真实盘口 replay。盘中滚撮全部采用静态盘面切片：

```text
当前时点 + 标的日/时段 + 剩余电量或目标
+ 成交进度 + 买卖盘状态 + 价格中枢
+ 剩余时间 + 风险约束
=> 判断盘口行为、风险和交易策略
```

合格答案需要覆盖盘口状态判断、判断依据、对我方缺口的影响、交易动作、量价边界、触发条件和风险提示。

## 外部信号口径

舆情、新闻、社交媒体、突发事件类 case 只评估分析路径，不评估实时检索覆盖率。合格答案应遵循：

```text
识别事件信号
=> 映射到供给、需求、成本、电网或交易行为
=> 推导价格、电量和策略方向性影响
=> 给出需要核验的数据
=> 给出风险和不确定性边界
```

禁止把传言当事实、编造政策文件号、编造实时数据或承诺收益。

## 评分口径

`scoring.yaml` 使用面向电力交易工作流可用性的 workflow-readiness 配置：

- `outcome`：结果质量，判断答案是否符合电力交易业务直觉、事实边界、风险约束和标准结论。
- `execution`：工具与执行，判断是否正确使用数据源、Skill、知识库或可用工具完成任务；缺私有或实时数据时，能否说明具体缺口和下一步取数动作。
- `orchestration`：编排协作，判断复杂任务拆解、多 Agent 协同、证据组织和边界收敛是否合理。
- `efficiency`：效率成本，判断时延和 token 使用是否处于可接受范围；效率只作为成本约束，不覆盖工作流质量判断。

评分链路固定为：

```text
metricResults score(0..1)
=> dimensionScore(0..100)
=> caseScore(0..100)
=> suite totalScore(0..100)
```

`scoring.yaml.dimensions.*.weight` 是默认维度配置；带有 `attributesJson.caseType` 的 active case 使用 `caseTypeWeights` 计算 case score。当前标准权重如下：

| caseType | outcome | execution | orchestration | efficiency | 设计原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| `data_query` | 50 | 25 | 20 | 5 | 问数类重点看结果与工具口径，同时保留数据边界、口径解释和补证路径。 |
| `business_answer` | 50 | 15 | 30 | 5 | 业务回答既要结论正确，也要体现证据组织、多步骤分析和可执行下一步。 |
| `scenario_analysis` | 45 | 15 | 35 | 5 | 外部信号和场景推演更依赖分析路径、拆解、证据链和风险边界。 |
| `strategy_advice` | 45 | 15 | 35 | 5 | 策略建议需要动作、量价边界、触发条件和协同推理。 |
| `robustness` | 50 | 10 | 35 | 5 | 缺信息和不确定场景重点看稳健结论、边界声明、补证动作和风险控制。 |

Suite 级 `totalScore` 是所有选中 case 的 `caseScore` 平均值。失败、超时、不完整、缺失结果的 case 仍保留在分母中，case score 为 0。

运行状态和评分资格需要分开理解：`status` 始终保留 Agent 或评测服务的原始运行态，不允许把失败运行重标为成功；`scoreEligibility` 用于说明该 case 是否有可评分的业务最终输出。非成功状态只有在同时满足以下条件时才会按同一套 `metricResults` 计分：

- `scoreEligibility=SCORABLE`。
- `finalOutputJson` 包含有用业务回答、数据边界说明、合理拒绝、能力不支持说明或需要补充信息的明确提问。
- 存在 score-bearing quality `metricResults`。

缺少 durable caseRun、空输出、错误兜底输出、提交前失败、校验失败、无有用输出的超时，以及未显式标记 `SCORABLE` 的非成功结果，仍按 0 分处理，并继续留在 suite 分母中。正式榜单发布资格由不可评分失败和不完整 case 阻断，而不是由 raw `FAILED` 标签本身阻断。

LLM Judge 或人工复核时，不要求固定措辞，也不要求在缺少证据的数据上给出精确数值。对于私有数据、实时行情、权限数据不可得的 case，只要 Agent 能明确说明缺失数据、解释该缺口对结论的影响、给出合理方向性判断、风险因素和下一步操作，即可获得中高分；如果只泛化要求“请上传数据”且没有业务分析或下一步动作，则只能视为低分或部分分。编造无证据数值、把传言当事实、承诺收益或回避用户目标仍应判低分。

数据边界类回答按以下口径校准：

- `0.75`：具体列出缺失字段、对象或时间范围，说明缺失数据如何影响交易结论，并给出保守下一步或可执行补证路径；可缺少完整数值结论。
- `0.5`：能识别任务并提出具体数据诉求，但缺少业务影响解释、临时方向判断或下一步执行策略。
- `0.25`：只泛化要求补充材料，未说明缺失项、业务影响或下一步动作。
- `0.0`：答非所问、编造关键事实、把不确定信号当确定事实或承诺收益。

`routing` 不再作为 active scoring dimension。当前 leaderboard 代码仍可能输出 `routingScore` 兼容列，该列不参与 ranking 解读。

## 标准结果指标

AgentX Bench 的主分保持克制，但结果数据必须完整。正式排名由 4 个维度和 7 个正式计分 metric 决定；标准报告、页面和榜单快照必须围绕 21 个结果指标解释本次分数是否完整、可信、可发布。

| 类别 | 指标 | 用途 |
| --- | --- | --- |
| 排名主指标 | `totalScore`、`outcomeScore`、`executionScore`、`orchestrationScore`、`efficiencyScore` | 官方 0-100 分数和维度对比。 |
| 运行完成度 | `selectedCases`、`completedCases`、`failedCases`、`incompleteCases`、`successRate` | 解释是否可评分完成；不可评分失败和不完整 case 不从 suite 分母剔除。 |
| 效率成本 | `avgDurationMs`、`tokenUsage` | 解释耗时和成本；主分只通过 `duration.normalized` / `token.normalized` 使用。 |
| 评分覆盖 | `qualityExpected`、`qualityScored`、`qualityUnavailable`、`qualityMissing`、`qualityNotApplicable`、`scoreConfidence` | 判断 Judge/人工质量指标是否足够完整，`diagnostic` 不进入正式榜单快照。 |
| 派生覆盖 | `derivedExpected`、`derivedAvailable`、`derivedMissing` | 判断 duration/token 是否足够支撑 efficiencyScore。 |

`leaderboardEligible` 是发布资格判定字段，不计入 21 个结果指标。正式榜单快照要求 run 完成、无不可评分失败或不完整 case、`scoreConfidence` 不是 `diagnostic`，并且 `leaderboardEligible=true`。`runtimeFailedCases`、`scorableNonSuccessCases` 和 `unscorableFailedCases` 是诊断字段，用于解释 raw status 和评分资格之间的差异，不新增为主分指标。

## Energent Swarm 数据口径

AgentX Bench 开源精简包只保留 Energent Swarm 自动评测入口。评测服务需要返回统一的 `caseResults + metricResults` 结构，再进入 AgentX Bench 评分。AgentX Bench 不处理登录、页面操作、外部 Agent API 适配或提示词编排。

推荐链路如下：

```text
同一批 case
=> Energent Swarm 输出答案、轨迹、工具调用、耗时和 token
=> 同一 LLM Judge 或人工复核规则转为 metricResults
=> score_run(scoring.yaml)
=> summary.json / leaderboard.csv
```

正式 ranking 不建议只使用 `score` 或 `totalScore` 这类单一人工总分。单一总分可以支持粗粒度排名，但不能解释 outcome、execution、orchestration、efficiency 的能力差异。正式对比应至少产出以下指标：

当前开源精简包只内置 Energent Swarm 自动评测入口，推荐参评对象以
`suite.yaml.recommendedSubjects` 为准：

| subjectId | subjectName | subjectType | agentMode |
| --- | --- | --- | --- |
| `energent-swarm` | Energent Swarm | `energent_swarm` | `multi_agent` |

| 维度 | 指标 | 评分重点 |
| --- | --- | --- |
| `outcome` | `reasoning.answer_accuracy`、`reasoning.task_completion_quality` | 结论正确性、任务完成质量、业务直觉和边界处理。 |
| `execution` | `skill.execution_quality` | 工具、数据源、知识库或证据使用是否正确。 |
| `orchestration` | `multi_agent.task_decompose_reasonability`、`multi_agent.collaboration_quality` | 任务拆解、多 Agent 协同、证据组织和复杂流程控制。 |
| `efficiency` | `duration.normalized`、`token.normalized` | 时延和 token 成本归一后的效率分。 |

导入的 `metricResults` 分数统一为 `0..1`：

```json
{
  "metricCode": "reasoning.answer_accuracy",
  "score": 0.82,
  "computeMode": "LLM_JUDGE",
  "reason": "结论方向正确，并说明了缺失数据边界。"
}
```

辅助解释指标应保留在报告和证据里，但默认不进入主分：

| 指标 | 用途 | 主分口径 |
| --- | --- | --- |
| `successRate` / `failedCases` | 解释稳定性和失败风险 | 失败 case 已通过 0 分进入分母 |
| `avgDurationMs` / `tokenUsage` | 解释效率和成本 | 原始值用于解释和 tie-breaker；主分只通过 `duration.normalized` / `token.normalized` 使用 |
| `toolCalls` / `toolSuccessRate` | 解释工具执行路径 | 可映射 execution，但需确认所有 Agent 可采集 |
| `traceSteps` / `retryCount` | 解释流程复杂度和重试成本 | 诊断字段，不建议直接加分或扣分 |
| `rawTrace` / `evidence` / `judgeReason` | 解释评分依据和支持审计 | 证据字段，不直接进入主分 |

新增主分指标必须同时满足：所有参评 Agent 可稳定采集、与能力强弱有明确关系、不会和现有维度重复计分。

## 发布排名口径

AgentX Bench 开源精简包应包含评测集、评分标准和 Energent Swarm 运行工具。
产品发布 ranking 建议作为带日期的快照单独沉淀，而不是作为永久在线榜单：

```text
published_results/power_trade_standard_v1/2026-06-xx/
  leaderboard.csv
  leaderboard.json
  run_metadata.json
  summaries/
```

每个发布快照必须说明数据集版本、评分版本、运行日期、参评 Agent、Agent 或模型版本、是否全量 43 条 case、Judge 模型或人工复核流程、失败 case 数和原始 summary/case_results 证据。

## 入库检查清单

- `caseId` 唯一且稳定。
- `inputJson.query` 包含 Agent 必须看到的上下文。
- `inputJson.teamName` 使用 `${subject.teamName}`。
- `expectedJson.expectedAnswer` 非空，并包含核心正确答案。
- `expectedJson.answerCriteria` 可评分，但不作为唯一正确答案来源。
- 所有相对日期已按 2026-06-23 改写。
- 无手机号、真实联系人等敏感信息。
- `attributesJson.category`、`caseType`、`difficulty` 完整。
- `suite.yaml.caseCount` 与 `cases.jsonl` 行数一致。
- `source_row_audit.jsonl` 覆盖全部非空源表行。
