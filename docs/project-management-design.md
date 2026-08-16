# 导演台多项目管理设计

> 状态：草案（待确认后实施）。本文描述如何把「单一长视频时间线」扩展为「多个可切换的长视频项目」。

## 1. 背景与根因

当前系统把「长视频项目」实现为一份**单例统一时间线文档**：

- SQLite 表 `unified_timeline` 主键固定 `singleton = 1`，`document` 列存一份 `UnifiedTimelineDraft`（version=3）JSON。
- 后端只暴露 `GET/PUT /api/timeline`，读写的都是这份单例。
- 顶栏「项目名」就是这份文档里的 `title` 字段（`project/patch` 只改这一个字段）。
- 前端自动保存 WAL 只有一个槽位（`director-web:v4:timeline-wal`），按「活动数据库身份」绑定一份待写时间线。

因此「新建一个长视频」只能覆盖这份单例，当前编辑内容随之丢弃。要支持多项目，需要在时间线文档之上引入「项目」实体，并让所有引用它的旁路（任务、素材级联删除、历史 take 账本、分段选择偏好、崩溃恢复 WAL）都按项目正确隔离。

## 2. 核心设计决策

1. **项目 = 一份拥有稳定 ID 的时间线文档**。项目表每行 (id, title, document, created_at, updated_at)，document 仍是现有 UnifiedTimelineDraft（不往文档里塞 project_id，避免污染创作文档）。
2. **「当前项目」是浏览器 UI 偏好，不是服务端状态**。多个标签页可以各自打开不同项目；服务端只在任务提交时按 URL/请求里带的 project_id 落账。
3. **素材库跨项目共享**。素材是 ComfyUI 工作区的资产，按 comfy_origin 登记，天然跨项目；级联删除改为解绑**所有项目**里的引用。
4. **历史 take 账本按项目隔离**（防御性）。同时保证新项目默认分段 ID 使用全新 UUID，杜绝跨项目稳定 ID 碰撞导致的 take 错用。
5. **任务记录打上 project_id**，任务抽屉按项目筛选；删除项目默认保留任务与 take（可审计），是否级联删除由用户在 §7 决定。

## 3. 后端数据模型

新增表（Database.initialize() 内 CREATE TABLE IF NOT EXISTS）：

    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,          -- UUID（服务端生成，客户端不可伪造）
        title TEXT NOT NULL,
        document TEXT NOT NULL,       -- UnifiedTimelineDraft version=3 JSON
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

迁移（幂等，跑在 initialize() 里）：

- 若 projects 为空且 unified_timeline 有行：把单例行复制为一个项目（id=新 UUID，title 取现有 title 或「未命名长视频」），并记 legacy_singleton_project_id 到 migration_notices 供旧端点兜底。
- jobs 表新增 project_id TEXT（可空）：现有 mode='timeline' 的任务回填为该迁移项目的 id；旧六模式任务留空，任务抽屉显示为「旧任务」。
- segment_takes 表新增 project_id TEXT（可空）：现有 take 全部回填为迁移项目的 id（它们都来自原单例时间线），之后 find_latest_segment_take 一律要求 project_id = ? 精确匹配。

**分段 ID 唯一性**：default_timeline_draft() 里写死的 timeline-segment-1 改为每次生成全新 ID（沿用 createLocalId 语义，后端用 uuid4 亦可）；新建项目时后端生成一份带新分段 ID 的默认文档。已存在的项目分段 ID 不动。

## 4. API 设计

新增项目级端点（project_id 从 URL 来，服务端校验存在性；config 仍随请求提交、服务端按验证后数据重新编译）：

    GET    /api/projects                     -> { projects: [{id,title,created_at,updated_at,segment_count}], active: null }
    POST   /api/projects                     -> 创建空项目（标题可空，默认「未命名长视频」）
    GET    /api/projects/{id}                -> ProjectSummary
    PATCH  /api/projects/{id}                -> { title } 重命名
    DELETE /api/projects/{id}                -> 删除（语义见 §7）
    GET    /api/projects/{id}/timeline       -> UnifiedTimelineDraft
    PUT    /api/projects/{id}/timeline       -> UnifiedTimelineDraft（沿用 validate_and_put_timeline 的资产校验锁）
    POST   /api/projects/{id}/compile        -> TimelineCompileRead
    POST   /api/projects/{id}/jobs           -> JobRead（落 project_id）

兼容层：保留 GET/PUT /api/timeline、/api/timeline/compile、/api/timeline/jobs，内部重定向到「迁移项目」（migration_notices 里的 legacy_singleton_project_id），等前端迁移完成后可下线。POST /api/jobs（旧六模式）不变。

任务列表 GET /api/jobs 增加可选 project_id 查询参数；传入时按 job.project_id 标注 current_project（并可供抽屉「当前项目」过滤），不传时行为与现在一致（全量、current_project=false）。

## 5. 前端状态与 WAL

- directorState 增加 projects: ProjectSummary[] 与 activeProjectId: string | null；时间线状态仍是 TimelineEditorState（project + 单一分段选择 + 当前检查器焦点 + 播放头）。实际可执行集合由 `selected ∩ enabled` 派生。
- **分段选择偏好**（workspacePreferences）使用 v2 envelope，按「数据库身份 + project ID」键控，并保存启用和停用段组成的完整选择；切项目时先恢复目标项目自己的偏好，无记录才默认全选，不能因两个项目复用相同 segment ID 而继承来源项目子集。
- **崩溃恢复 WAL 改为按项目分槽**：升级到 v5，键 director-web:v5:timeline-wal:<projectId>。保留 v4 单槽的隔离/迁移动线（旧键只归档不重放），复用现有 owner_id/quarantine 机制防跨标签页错清。
- **切项目流程**（关键，必须 fail-closed）：
  1. 若当前项目有未确认 WAL，先 force-flush（await 服务器确认 + 清槽）再切；
  2. 用 latest-wins 世代读取目标项目槽位 WAL 或 GET /api/projects/{id}/timeline；选择当前项目、发起更新的切换、开始生成或预检都必须使旧请求失效；
  3. 目标读取期间当前项目仍可编辑，因此权威交接前必须再次 force-flush，并复查切换世代与运行 intent；
  4. 在同一同步交接中恢复单一分段选择、播放头归零、清空预检报告与历史 take 的 current-snapshot 展示。
  任一步同步失败或请求已过期都保持当前项目不变，不能让迟到响应取得权威。

## 6. UI：项目切换器（顶栏）

顶栏现「项目名」位置改成下拉/弹出：

- 当前项目名 + 展开图标；下拉列出全部项目（标题、更新时间、分段数）。
- 「新建项目」：POST 创建 → 切到新项目。
- 「重命名」「删除」（删除二次确认）。
- 切换前自动 flush 当前项目（§5 流程），失败则阻断并提示（沿用 revision 阻断语义）。
- 任务抽屉「当前项目」过滤改为基于 project_id；旧任务（无 project_id）显示为「旧任务」分桶。

## 7. 删除语义（待确认，见问题）

默认方案：**删除项目只删项目与时间线，保留任务历史与 take**——任务变为「旧任务」仍可查看/下载，take 成为孤儿行（无害，不再被查找）。可选方案：级联删除该项目全部任务（take 保留，因 take 是成片不是生成缓存）。

## 8. 跨切面影响清单（实施时必须逐项处理）

| 切面 | 现状 | 改动 |
| --- | --- | --- |
| unified_timeline 单例 | 唯一权威 | 迁移为 projects 首行，旧端点兜底 |
| get_timeline/put_timeline | 单例读写 | 改为按 project_id 读写 |
| validate_and_put_timeline 资产校验锁 | 只锁单例 | 按目标项目锁 |
| delete_asset_if_unused(cascade) | 只解绑单例+六草稿 | 解绑**所有项目** + 六草稿 |
| segment_takes 账本 | segment_id+指纹+origin | 增加 project_id 维度并精确匹配 |
| jobs.project_id | 无 | 新增列，任务提交落账，抽屉按项目过滤 |
| current_project / current_snapshot | 与单例时间线比 | 与「活动项目」比（server 收 project_id 或 client 用 project_id 判断） |
| 历史「加载任务来源项目」 | 覆盖单例 | 改为「另存为新项目」或询问覆盖 |
| 前端 WAL | 单槽 v4 | 按项目分槽 v5 |
| 分段选择偏好 | v2 按数据库身份 + project ID 隔离 | 切项目同步恢复，停用段也保留 |
| default_timeline_draft 固定分段 ID | timeline-segment-1 | 每次生成全新 ID |

## 9. 分阶段实施

1. **后端**：projects 表 + 迁移 + 项目 CRUD/时间线端点 + take/job 加 project_id + 资产级联遍历全部项目（含单测）。
2. **前端数据层**：ProjectSummary 类型、API client、WAL v5 分槽、directorState 加 projects/activeProjectId。
3. **前端 UI**：顶栏项目切换器 + 新建/重命名/删除 + 任务抽屉按项目过滤。
4. **联调与回归**：切项目 flush、旧端点兼容、级联删除、take 隔离、双标签页、重启恢复。

## 10. 测试要点

- 迁移幂等：空库 / 有单例 / 已迁移，各跑一次 initialize() 结果一致。
- 切项目前有未确认 WAL：flush 成功/失败两种路径都不丢数据、不越权覆盖。
- 两个项目各自同名分段 ID（手工构造）不会互相复用 take。
- 级联删除素材：从所有项目解绑，任一步失败整笔回滚。
- 删除项目后：任务仍可查看，current_project 正确变 false。
- 双标签页同一数据库、不同项目：WAL owner 不互相清槽。

## 11. 实现状态（已落地）

- 后端与前端已按本设计实现，全量测试通过（backend tests + frontend 391 项）。
- **关键实现偏差**：§3 草案描述“把单例复制成 projects 首行”。最终实现改为——
  「默认项目」（id=default）直接映射到原 `unified_timeline` 单例行（单一权威，避免双份副本分叉）；
  只有新建项目才写入 `projects` 表。
- 项目标题 = 时间线文档 `title` 字段（单一权威），`projects.title` 列作为列表冗余并在保存/重命名时同步。
- 顶栏项目切换器用原生 `<select>`（切换 + 新建），删除用独立按钮，重命名复用原顶栏标题就地编辑。
- 任务抽屉「当前项目」仍按严格 timeline 相等（现有契约），列表端点新增 `project_id` 查询参数使比较按活动项目作用域。
- 历史「加载任务来源项目」已改为「另存为新项目」：通过 `POST /api/projects/import` 原子创建新项目并切换，保留来源片段的稳定 ID 与显式分段选择，并立即写入新项目自己的选择偏好，不再覆盖当前编辑中的项目。
