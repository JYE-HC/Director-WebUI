# 统一长视频时间线设计

## 产品目标

导演台的工作对象是一部影片，而不是某一种生成模式。编辑器只让每段选择 FL2VA 或 Ref2VA
模型族；T2V、I2V、FL2V、R2V、V2V、RV2V 是后端根据该段 typed 素材推导出的执行配方。
同一条主轨可以按顺序混合使用两个模型族和六种配方。

网页保存的统一时间线文档是唯一权威数据。前端不维护或提交 ComfyUI workflow；后端按
版本化固化模板直接生成 API prompt。

## 页面结构

- 左侧资产库：显示当前 ComfyUI 工作区登记的图片、视频和音频。支持类型过滤、网格密度、
  上传、拖放、排序，以及从素材库批量移出。资产身份使用服务端 ID；路径只作为展示信息。
- 中央主预览：显式素材选择优先；播放时间线时，按稳定 segment ID 使用最新任务中非歧义的
  `segment_results` 候选。没有候选时才显示源视频或 slate，不能把参考素材伪装成生成成片。
- 右侧实时进度：显示活动分段、阶段、片段进度、采样 step/total，并允许在 ComfyUI 支持原子
  取消时终止当前任务。后端只接收 ComfyUI 标准 WebSocket 的 metadata-bearing event-4
  PNG/JPEG 预览，严格绑定登记过的 child prompt 与 sampler node；最新帧仅进入有 TTL/总量上限
  的进程内缓存，不写 SQLite，旧式无 metadata 帧和超过 2 MiB 的帧直接丢弃。
- 操作栏：插入、分割、复制、合并、删除；分段复选框是唯一运行范围，全选即运行全部启用段。
- 主时间线：唯一主轨，无重叠。首次绑定或拖入 Ref2VA 源视频时，以服务器探测的完整时长铺到主轨，
  并显示等距关键帧画面带；长源片是可自动保存、可播放和可分割的编辑状态，但超过原生 H3 512 帧时
  必须先用播放头、均分或智能分割切段才能生成。另支持插入空 FL2VA 生成分段、拖动排序、
  Ctrl/Command 切换选择、Shift 范围选择及键盘删除。
- 分段检查器：只编辑当前选择。模型族切换会重建该族允许的字段，禁止隐藏的另一族参数串入。
  图片、视频、音频参考槽各自编号；提示词标签自动完成只显示当前分段已经绑定的槽位。

全局设置、任务面板、系统设置和主题仍位于页面壳层，不占用时间线工作区。

## Canonical 数据模型

统一文档 `version=2`，包含项目标题、输出规格、FL2VA/Ref2VA 两套采样参数、参考图尺寸、
音频/导出策略、连续性设置和有序 `segments[]`。每个分段都有稳定 UUID、启用状态、标题、
`mode=fl2va|ref2va`、时长和提示词，再使用严格判别联合保存该模型族的素材：

- FL2VA：可空的 `first_image` 与 `last_image`。无锚点推导 T2V，仅首图推导 I2V；只要尾图存在
  （尾图单独或首尾都有）就推导 FL2V。
- Ref2VA：显式区分一个可空 `source_video` 与独立图片/音频/视频参考网格。仅独立参考推导 R2V，
  仅源视频推导 V2V，源视频加任意独立参考推导 RV2V。空段可保存编辑但不可提交。

官方 conditioning 的能力上限按素材类型分别计算：参考图片最多九张、视频总计最多三路、
独立参考音频最多三条，不存在跨类型共享的总数上限。源视频占第一路视频，独立参考视频随后排列，
因此有源视频时独立视频最多两路，无源视频时最多三路。

`<Picture N>`、`<Video N>`、`<Audio N>` 都是分段局部标签，不是资产库序号。FL2VA 图片标签按当前
锚点压密派生：仅首图/仅尾图均为 `<Picture 1>`，首尾俱全时依次为 `<Picture 1>/<Picture 2>`；它们
不是 Ref2VA 的持久化图片 slot。Ref2VA
主视频固定解释为 `<Video 1>`，独立视频从下一标签开始；启用源音轨参考时它同时占 `<Audio 1>`，独立音频标签顺延一位，
但内部 slot 不变，三个独立音频名额也不会减少。静音或历史未知音轨的视频不能启用。I2V/FL2V 锚点不占 R2V 图片槽。

## 执行语义

统一时间线提交为一个父任务，每个所选段编译为一个固化原生 child prompt。关闭接续时，后端按
Standard→RayLight、FL2VA→Ref2VA 分组，并在组内保持时间线顺序；开启接续后则按时间线逐段提交，
同次运行依赖边严格等待并动态绑定，服务端已经解析并绑定的历史 take 边不等待旧任务；不再按后端或
模型族重排。同族同后端的各 prompt 重复完全相同的
loader 节点 ID/输入，使 ComfyUI 跨 prompt 复用缓存，同时把失败、取消和 decode 缓存限制在单段。
浏览器不能传入任何节点或连线。

标准子图只使用 ComfyUI core/官方 extras 节点；自动推导的 H3 LoRA loader 和多卡 GPU 池派生的
RayLight 是两类精确白名单例外。`MiniMaxH3Director` 节点不在执行路径。六种配方由官方
`MiniMaxH3ImageToVideo`/`MiniMaxH3ReferenceToVideo` 的 typed 输入表达。

“生成所选”只提交稳定 ID 指定的段，“分段导出”返回严格对应的分段文件，“组装全片”必须确认
每个启用段恰有一个本次任务输出，再由后端 ffmpeg 统一规格并拼接；不得用旧缓存、源视频或灰色
占位偷偷补洞。任务读取通过 `segment_results[]` 返回稳定 segment ID 到本次 child/output 的非歧义
映射，供前端把候选 take 放回正确分段；完整组装片仍保留在父任务 `outputs[]`。父任务保存子 prompt、
模板版本和设置快照，支持逐段进度、取消和错误定位。失败分段不会抹掉其他已成功的候选，父任务等
所有 child 终态后再报告部分失败。

后台对账以父任务为批次：每轮一次 queue snapshot、一次受限 bulk history；只有 queue 成功时才对
bulk 遗漏项按轮转窗口精确补查，每轮最多 16 项。queue 暂时不可用时只接纳 bulk 中的确定结果，
不对最多 128 个 child 逐一请求，也不从“未出现”推断终态。WebSocket 补充原生节点执行阶段、采样步数和受控实时帧，
queue/history 仍是生命周期权威。浏览器的任务列表与详情 GET 只读 SQLite，不代理触发对账。

连续性使用官方 `MiniMaxH3AddGuide` 做显式尾帧条件接续，不宣称等同于旧 Director 的跨段 AV latent
handoff。首个启用段以及带 `first_image` 的 FL2VA 段是锚点重置；其他段依赖完整启用时间线中紧邻的
前一个启用段；T2V、FL2V、Ref2V 以及 FL2VA/Ref2VA 可以混排接续。若前驱也在本次选择中，后端在其 exact history
成功并确认声明的唯一 `SaveVideo` output 后动态绑定；若未选择前驱，后端按当前 endpoint、稳定
segment ID 和输出几何规格（宽高、FPS、H3 可见帧数）解析最新成功 take，并在 runtime 预检前绑定，
浏览器不能提交 take 或路径。提示词、生成配方、参考素材与推理参数不参与该匹配。
两种来源都读取最后 N 帧（5/22/39/56），以
`MiniMaxH3AddGuide(frame_idx=0)` 约束后段开头，内部采样 `align(F+N)` 帧，再在保存前只截取
`[N,N+F)`。因此前置接续帧和 alignment tail 都不会进入成片，单段可见 F 帧及全片总时长与关闭接续时
相同；生成音频按同一区间接续和裁剪，源音频/静音仍服从当前段策略。前驱失败、被 ComfyUI 外部取消
或没有唯一持久输出时，同次运行中未提交的传递后继以依赖失败收口，不把未绑定图交给 ComfyUI；历史
take 缺失、输出几何不兼容、endpoint 不同，或生成音频却没有音轨能力时，在创建任务前以 422
失败。用户取消父任务则停止续提并保留取消语义。

Standard 模型复用依靠稳定 loader 输入。RayLight 默认按 family + model + LoRA + GPU pool + topology + sigma shift
完整 key 常驻并跨分段/任务复用 CUDA 权重；不兼容 Ray key 或 Standard 会先等待 RayKill 屏障成功，再以
递增持久 epoch 创建新池。FL2VA 与 Ref2VA 可在同一组 GPU 上顺序运行，无需重启 ComfyUI，也不依赖 OOM
自动卸载。明确选择任务后释放时，每个 sampler 后释放 Ray worker 权重，下一次 Ray 任务需要重新加载。
当前安装版 release 模式保留 ModelPatcher/actor handle，只卸载 CUDA 权重，因此同 key 可沿用 epoch，
下次 sampler 再载入 CUDA；切到 Standard 仍先 shutdown Ray cluster。持久 tail/taint 状态使取消、失败和
后端重启后的不确定池先走屏障。
预发布 v1 账本缺少完整 loader chain 时，迁移保留旧 epoch 并把 actor 状态标为未知；在 Director 先成功
用新 epoch 提交一次 RayLight 任务、由 initializer 显式替换旧池前，Standard 提交失败封闭。
这些保证只适用于 Director 串行提交的任务。ComfyUI Web 手工 workflow 不受其 endpoint 锁调度，
干扰也不一定能被 Director 检测，因此不得混入 Director 管理的 Ray 常驻序列。

## 素材与删除边界

资产记录绑定上传时的 canonical ComfyUI origin。切换 endpoint 后，旧资产不会被当成新服务器
同名文件。网页“从资产库移除”只删除 Director 的本地登记，不越权删除远端文件；仍被时间线引用的
资产默认返回 409。显式 `cascade=true` 会在同一 SQLite immediate transaction 中解除统一时间线
和六份旧草稿的 typed 引用、压密 reference slots、同步重写分段提示词标签、让后端按剩余素材重算配方并
重新严格校验，然后才删除本地登记；任一步失败整笔回滚。返回的 usage 清单是审计记录，远程输入和
生成输出仍保留。删除素材始终保留用户选择的 FL2VA/Ref2VA 模型族；FL2VA 随剩余锚点推导 T2V/I2V/
FL2V，Ref2VA 随源和独立参考推导 R2V/V2V/RV2V。最后一项 Ref2VA 素材被删后，该段保留为未完成
编辑状态且提交失败封闭，不越权切到 FL2VA。若删除源视频使 `audio_mode=source` 不再成立，则归一为
`generate`。旧六模式草稿只解除其 typed 引用，不冒充统一时间线推断模式顺序。未来如支持远端删除，
要使用单独、明确且可审计的命令。

## 迁移与兼容

- 保留旧 `/api/drafts/{mode}` 与旧任务读取接口，避免升级后历史任务失联。
- 单例 `/api/timeline` 的 v1 六模式分段严格验证后迁移到 v2 两族形状；非法旧跨模式字段仍拒绝。
- 旧六份草稿不自动拼接，避免把无关创作误合成一部影片。
- 首次升级只把一个有实际内容的旧草稿复制为统一时间线候选；存在多份有效旧草稿时保留原数据，
  由用户选择导入顺序。
- 浏览器旧 localStorage 采用同样的非破坏迁移；服务端时间线确认前不覆盖本地尚未完成自动同步的编辑。

## 尚未覆盖的剪辑层能力

- undo/redo、吸附、ripple 编辑、缩放、轨道锁定、转场和字幕/音频轨。
- 候选的显式 accepted/stale 生命周期、版本比较和批量回退；当前已能按 segment ID 显示最新候选，
  但不会把“最新”自动持久化为用户已接纳版本。
- 项目级人物、服装、场景、镜头语言和 speaker continuity bible。
- 分段音频 trim/fade/crossfade、响度归一化和明确的缺音策略。
- 显式 accepted/rejected 段级重试状态；当前“生成所选”会自动复用同稳定 segment ID 与输出几何的直系前驱最新成功 take，
  但仍不会替用户把某一候选标记为已接纳版本。
- 剪辑层 crossfade/转场；它与当前官方 H3 guide 条件接续及旧 latent handoff 都是不同能力。
