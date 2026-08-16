# 后端固化的原生工作流执行架构

本文档定义统一长视频时间线如何调用 ComfyUI。它是运行边界，不是前端可编辑的工作流格式。

## 不变量

- 浏览器只提交严格校验的时间线、素材 ID、分段选择和系统设置，永不提交 `prompt`、
  `workflow`、`class_type` 或节点连线。
- 后端按版本化模板构造 ComfyUI API prompt；任务提交时重新编译，不能提交一次预检返回的图。
- 标准执行只允许 ComfyUI core 与官方 `comfy_extras` 节点。
- 自定义节点只有两类例外：后端从受支持 H3 LoRA 文件名确定性推导的加载器，以及 GPU 池自动启用的
  RayLight 多卡执行节点。
- 新时间线执行路径不得包含 `MiniMaxH3Director`，也不得调用 Director 插件的上传、探测、分镜或进度接口。
- 每个生成任务保存设置快照、模板版本、分段 ID 与后端选择结果，作为审计和再次提交的依据。

## 父任务与子图

一次时间线提交创建一个 Director 父任务，并为每个所选分段创建独立 child prompt：

- 分段 `mode=fl2va|ref2va` 直接选择模型族；后端按 typed 素材另行推导六值 `recipe`；
- FL2VA 的无锚点/仅首图/存在尾图分别推导 T2V/I2V/FL2V；
- Ref2VA 的仅独立参考/仅源视频/源视频加任意独立参考分别推导 R2V/V2V/RV2V；
- 同族同后端 prompt 使用稳定一致的扩散模型、CLIP、VAE loader 节点 ID/输入，由 ComfyUI 跨 prompt 复用缓存；
- 每个 prompt 只包含一个分段的 conditioning、采样、解码和保存链，分段失败/取消不抹掉其他 take；
- 关闭连续性时保持 Standard→RayLight、FL2VA→Ref2VA 的稳定分组顺序；开启连续性时所有 unit 严格按
  时间线顺序执行，不能跨越前驱依赖重排；
- 子任务输出按稳定 segment ID 回填，父任务成功后由后端按时间线顺序用 ffmpeg 合成长片。

父任务负责按分段复选集合运行、取消、子任务聚合、输出缺口检查和最终拼接。ComfyUI 只负责执行受控生成图，
不承担产品级时间线状态。

每个 child UUID 在调用 `/prompt` 前就写入 SQLite，并作为 ComfyUI `prompt_id` 提交，缩小后端崩溃时的
匿名任务窗口。取消只使用当前 ComfyUI 原生、按 ID 原子执行的 job-cancel API；旧版本若无该端点则
失败封闭，不回退到可能全局中断错误任务的 `/interrupt`。HTTP client 断连不会取消已经持久化的
提交批次；正常关机或硬重启会把中断的 `preparing/submitting` 父任务及已绑定 prompt 恢复到可终结的
取消流程。启动扫描同时按非终态 child 的 `submitting/cancelling_during_submit` ownership marker 查找，
因此即使 parent 已停在任意 `cancelling` stage 也不会漏掉；未提交 child 明确标记终态，不能永久卡住。

## 标准原生模板

共享模型链：

```text
UNETLoader -> SelectModelDevice -> [允许的 LoRA loader] -> MiniMaxH3SigmaShift
CLIPLoader -> SelectCLIPDevice
VAELoader(video/audio) -> SelectVAEDevice
```

每段采样链：

```text
MiniMaxH3ImageToVideo | MiniMaxH3ReferenceToVideo
  -> BasicGuider
MiniMaxH3SigmaShift -> BasicScheduler
KSamplerSelect + RandomNoise + guider + sigmas + latent
  -> SamplerCustomAdvanced
  -> VAEDecode + VAEDecodeAudio
  -> CreateVideo -> SaveVideo
```

开启接续后，非锚点重置段在 conditioning 与保存链中增加官方节点：

```text
前驱 child exact-history 中声明的唯一 SaveVideo output
  -> LoadVideo(file="... [output]")
  -> GetVideoComponents
  -> ImageFromBatch(batch_index=-N, length=N)
  -> MiniMaxH3AddGuide(frame_idx=0)

VAEDecode(sample=align(F+N))
  -> ImageFromBatch(batch_index=N, length=F)
  -> CreateVideo -> SaveVideo
```

这里 F 是该段原本的 `17k+5` 可见帧数，N 只允许 5/22/39/56；`align(F+N)` 会再产生对齐尾帧。
保存节点只收到 `[N,N+F)`，因此前置 N 帧和 alignment tail 都不会进入分段文件或最终长片。生成音频模式
用 `TrimAudioDuration` 同时取得前驱尾部音频 guide，并在解码后截取 `N/24` 起、持续 `F/24` 的可见音轨；
源音频模式继续使用当前 Ref2VA 段自己的完整源裁剪音轨，静音模式不接音轨。依赖 FL2VA 段的
`last_image` 仍接入 `MiniMaxH3ImageToVideo.last_frame`，让 Qwen 图文输入保留对应的 `<Picture N>`
视觉块；同时另用同一张图的 `MiniMaxH3AddGuide(frame_idx=N+F-1)` 锚在最后一个可见帧。节点在采样
末尾产生的同图隐式锚点位于随后会被裁掉的 alignment tail。

配方差异只存在于后端模板的 conditioning 输入：

- T2V：`MiniMaxH3ImageToVideo`，不接首尾帧；
- I2V：`LoadImage` 接 `first_frame`；
- FL2V：首帧、尾帧或两者始终接入 `MiniMaxH3ImageToVideo` 形成压密的 `<Picture N>` 图文标签；依赖
  接续且带尾图时，再按上文用同图 `MiniMaxH3AddGuide` 锚到 `N+F-1`；
- R2V：图片、音频、视频分别接入 `MiniMaxH3ReferenceToVideo` 的 autogrow 输入；
- V2V：`LoadVideo -> Video Slice -> GetVideoComponents` 的画面固定作为 `<Video 1>`；
- RV2V：V2V 主视频再叠加独立图片、音频或视频参考。

Ref2VA 始终显式区分 `source_video` 与独立 `reference_videos`。官方节点分别支持最多九张参考图片、
三路视频和三条独立参考音频，各类型之间不共享总容量。源视频占 `ref_video_0`，独立视频随后接
`ref_video_1..`；所以存在源视频时最多再接两路独立视频。分段开关 `source_audio_as_reference` 可把同一个 `GetVideoComponents.audio` 接到
`ref_video_audios.ref_video_audio_0`，由官方节点把它作为配对 `<Audio 1>`；独立音频随后
从 `<Audio 2>` 开始，但仍可使用全部三个独立音频槽。该开关只允许服务器 ffprobe 持久化 `has_audio=true` 的源视频，历史未知值
按 false 处理；R2V 参考视频不会自动配音轨。`audio_mode` 仍独立决定最终 `CreateVideo` 使用生成音频、
源音频或静音，保留源音频同样要求 `has_audio=true`。

MiniMax H3 原生模板使用 `BasicGuider` 且产品不提供负面提示词，因此不存在可调的推理 CFG。参考视频在上传后由
后端生成不可变 24fps 代理；工作流只读取代理。Picture、Video、Audio 三类槽位必须分别连续为
`0..N-1`，否则提交失败，防止 autogrow 的展示顺序与提示词编号错位。

空 Ref2VA 是允许保存的未完成编辑状态，但 compile/submit 必须失败；脱敏计划同时返回模型族
`mode` 和实际六值 `recipe`，浏览器仍不能获得或回传工作流图。

## 标准与 RayLight 的确定性选择

FL2VA 与 Ref2VA 各自保存独立执行配置。标准模式使用单个 `device`；RayLight 使用 ComfyUI
逻辑 GPU 列表。GPU 池是执行后端的唯一权威，规则固定为：

1. 一张已选 GPU 编译 Standard 原生模板；
2. 两张或以上已选 GPU 编译 RayLight 模板；
3. 历史 `backend=standard|raylight` 只为兼容旧 API/数据库而接受，保存和启动迁移统一写回 `auto`，
   编译器始终忽略该旧值；
4. 派生 RayLight 所需节点、GPU 或拓扑不满足时直接预检失败，绝不静默退回单卡；
5. H3 的 `cfg_degree` 固定为 1，`GPU` 必须等于 `GPU_SELECT` 的数量，并满足并行度乘积约束。

RayLight 模板继续使用官方 H3 conditioning、VAE 解码和 SaveVideo，只替换模型加载、sigma、guider、
scheduler 与 sampler 为精确 allowlist 中的 RayLight 节点。所有逐段 RayLight child 都按既定 family/
timeline 顺序执行，避免同一 ComfyUI 进程中 `ray.shutdown()` 让不同 actor 池互相破坏。后端还会按
endpoint 串行提交每个父任务的全部 child，避免并发 HTTP 请求把两个父任务交错。
这里的串行不是一次把整批 prompt 排进 ComfyUI：每个 Ray 生成 child 必须取得 exact history 成功终态，
才允许提交下一 child。失败或 ComfyUI 端外部移除会 taint 当前池；父任务未被用户取消时先执行 RayKill，
再以新 epoch 继续尚未提交的分段并保留已成功 take。用户取消则立即停止续提。创建请求在预检、持久化和
endpoint 顺序票据登记完成后返回 `preparing`，长时间推理由受管 dispatcher 继续执行。

默认策略是 `raylight_residency_policy=keep_until_switch`。旧的按模型族常驻设置会一次性
迁移到该策略。后端用完整 loader chain（包含 family/model/LoRA/GPU pool/topology）及会原地修改 worker
模型的 sigma shift 构成 runtime key；同 key 连续分段和
后续父任务共享持久 epoch 与 worker CUDA 权重。切换 key 或 Standard 时，在 endpoint 提交锁内先排入
`RayKill` 并等待 exact history 成功，再递增 epoch 后提交目标 prompt。这样 A→B→A 的第三个 A 不会命中
第一个 A 已被 shutdown 的 actor handle。明确选择 `release_after_sampling` 时模板固定
`clear_vram_after_sampling=true`，每次采样后释放 worker 权重；采样前与采样后的 Comfy driver 清理都服从
初始化器中固定的 `driver_cleanup_policy=ray_devices`，只释放本地 `GPU_SELECT` 对应逻辑卡上的 driver 模型，
不再清空整个 ComfyUI 可见设备集合。放在非 Ray 卡上的 CLIP/Video VAE/Audio VAE 继续由 ComfyUI 自动管理，
不会被 Ray 清理路径强制卸载，但仍可能因显存/内存压力、模型切换、Free Memory 或重启而被淘汰，因此只是
尽力保温而非绝对常驻。辅助模型与 Ray 共用卡时，`release_after_sampling` 会释放 worker 并在同一卡池执行
定向 driver 清理，为后续 VAE 解码或其他负载安全腾出空间；`keep_until_switch` 则不能承诺跨进程权重与辅助
模型同时常驻。下一次任务仍沿用同 key actor epoch：当前安装版不会置空 ModelPatcher/active key，而是
unpatch/offload CUDA 权重，所以下一次即使命中 loader 缓存，也会由 sampler 把同一模型重新载入 CUDA。
它仍会追踪当前 Ray cluster，切到 Standard 时也必须经过完整 loader chain → `RayKill` 屏障；升级 RayLight
后需重新审计该契约。
运行状态同时持久化 queue-tail prompt 与 taint：失败、取消、重启后无法确认、或 queue/history 契约异常
都不能直接复用；屏障只有 exact history 明确成功后才允许提交新目标。屏障 child 没有 segment，不计入
分段进度/输出/公开 child 数，但保留在 SQLite 中供定向取消和启动恢复。
预发布 v1 运行账本没有可验证的完整 loader chain；迁移会保留其 epoch、把旧 actor 池标成未知，并在一次
Director RayLight 任务以新 epoch 显式重建池之前阻止 Standard 提交，不能把未知旧池当成已经释放。
该串行化只约束 Director 自己提交的 prompt；ComfyUI Web 手工 workflow 绕过 endpoint 锁与状态账本，
不享受自动切换保证，且其干扰不一定能被 Director 检测，因此不得与 Director 管理的 Ray 常驻序列混跑。
升级会为旧常驻设置完成一次映射并写入持久 marker；用户明确保存的 release 策略不会在后续重启时被改写。
历史 job 的 settings snapshot 是执行审计，始终保留当时的 release 值，不随 live settings 迁移。

LoRA 路由自动且失败封闭。Standard 只认可当前经过审计的精确 basename：
`minimax_h3_turbo_v4_step600(_ema)` 使用 `MiniMaxH3TurboLoRA`，两套 4/8-step
`*_10ErosMax_beta1_pruned_compat_v001_T8` 使用 `LoraLoaderBypassModelOnly`，对应
`*_comfyui_bf16` 使用 `LoraLoaderModelOnly`；RayLight 一律使用 `RayLoraLoader`。相似后缀、未知命名、
缺节点或错误 provenance 均在 `/prompt` 前失败，不猜测或回退。旧 `lora_loader`/`lora_low_vram`
只为读取旧设置而保留，live 设置归一为 `auto/false`，编译器不让它们成为隐藏行为开关。原生时间线
v1 固定关闭 RayLight FSDP/CPU offload；当前结构验证覆盖 U2 模板，但没有足够证据证明 FSDP actor
路径在逐段采样后完整释放 CUDA 引用，因此不能把节点表单中的选项当成已支持能力。

## 安全与能力预检

后端在 `/prompt` 前执行以下检查：

- 图中每个 `class_type` 都属于对应模板版本的精确 allowlist；
- 标准节点的 `python_module` 必须与已知 core/`comfy_extras` 来源一致；输入连线由服务端模板
  固化，并在 ComfyUI 接受 `/prompt` 时由其原生 validator 再校验；
- LoRA 和 RayLight 例外必须来自精确允许的节点集合，且只在设置实际启用时出现；
- 用户素材输入路径从数据库中的不可变 asset 记录生成；客户端不能提供路径或 SaveVideo 前缀；
- 接续 unit 保存前驱 segment ID、N、目标 `LoadVideo` 节点 ID 与
  `source=same_run|historical_take`。同次运行依赖在前驱成功前保持不可提交占位，后端只能从该 child
  声明的 SaveVideo node 取得恰好一个 history `type=output` 描述再纯函数式绑定。未选前驱则只能由
  服务端按当前 endpoint、稳定 segment ID 与输出几何（宽高、FPS、H3 可见帧数）解析 durable take，
  并必须在 runtime 预检前完成同样的
  安全绑定；浏览器不能传 take ID、输出描述或路径；
- 模型文件、LoRA、逻辑 GPU、并行度、素材类型、24fps 代理和连续槽位均有效；
- 节点链接必须指向图内节点，节点数和分段数受上限约束；
- `/compile` 只返回脱敏的分段计划、后端选择、节点类型与 policy，不返回可执行 prompt。

`native_timeline.continuity` 是独立可选能力：只有 `MiniMaxH3AddGuide`、`ImageFromBatch`、
`TrimAudioDuration` 都存在且 `python_module` 与官方来源精确一致时才报告 true。缺失或同名自定义覆盖只会
阻断接续图，不应把关闭接续仍可执行的基础原生时间线错误报告为不可用。

## 连续性与进度

当前连续性是官方 H3 guide 条件接续，不是旧 Director 的跨段 AV latent handoff，也不是剪辑层
crossfade。首个启用段以及带显式 `first_image` 的 FL2VA 段是锚点重置，不读取前段输出；其余段依赖
时间线上紧邻的前一个启用段，允许 T2V、FL2V、Ref2V 与 FL2VA/Ref2VA 混排。前驱在本次运行中时，dispatcher
只有等其 exact history 成功且声明的 SaveVideo node 产生恰好一个持久 output 后才动态绑定并提交后段；
前驱失败、被 ComfyUI 外部取消或移除、输出缺失或歧义时，所有尚未提交的同次运行传递后继标为
`status=failed, stage=dependency_failed`。前驱未选择时，compile/submit 各自重新从无外键的
`segment_takes` 账本解析当前 endpoint 上同稳定 segment ID 且宽高、FPS、H3 可见帧数一致的最新成功 take；
历史依赖在预检前已经绑定，不进入 same-run 等待或失败传播图。缺失、几何不兼容或
`audio_mode=generate` 所需音轨能力不足均在建任务前返回 422。提示词、生成配方/模型族、素材引用、模型、
LoRA、采样、Seed、GPU、接续、标题、启用状态与顺序都不参与匹配；任务删除保留 take。用户取消整个父任务时 cancellation ownership 优先，dispatcher
立即停止续提，未提交后继按取消语义收口而不是误标依赖失败。

接续只扩大内部采样上下文，不改变输出时长：每段计划同时记录 `visible_frame_count=F`、
`sample_frame_count=align(F+N)`、`continuity_context_frames=N`、alignment tail、前驱 ID 和
`anchor_reset`，并显式标记 `continuity_source` 与可选 opaque `historical_take_id`；保存文件仍恰好 F 帧，
完整长片仍按各段原可见帧数求和，计划不暴露历史输出路径。

执行阶段使用 ComfyUI 标准 WebSocket 的 `execution_start` 与 `executing(prompt_id,node)` 事件，按服务端
固化 prompt 中的节点类型显示模型加载、条件构建、采样准备、VAE 解码、封装和保存。RayLight worker
通常不会为加载、条件构建、解码和保存上报 `step/max`，这些阶段只保证“当前节点阶段”可见；但 ComfyUI 主进程仍会为 RayInitializer、RayUNETLoader、Ray guider/
scheduler 与 XFuser sampler 等图节点发送 `executing`，因此阶段可见。采样器另用
`progress(value,max,node)` 提供精确 step/total；非采样阶段没有标准总量，只显示粗粒度百分比里程碑，
不能伪造连续进度。后端保存每个 child 节点到 segment ID 的映射并聚合到父任务；后台每轮对账对每个父任务只取一次 queue，
仅对离队 child 读取受限 bulk history；queue 可用时每轮最多对 16 个缺失条目按 prompt ID 补查，
queue 不可用时不做逐段补查也不从“缺席”推断取消，避免异常 endpoint 上的请求风暴。
全片 ffmpeg assembly 只由后台 reconciler 启动；HTTP 任务列表/详情对 timeline 与历史 legacy job
都只读 SQLite，不请求 queue/history，因此黑洞 ComfyUI 不会冻结任务面板。分段输出 URL
直接使用已经持久化的 child 输出映射，不通过读取请求触发拼接。重启时 lifespan 在对外可用前
仅使用 SQLite 替换死进程的 submission ownership；定向 cancel 由启动后可取消、有界批次的恢复
worker 执行。cancel 未确认时保留 recovery marker 重试，不能以瞬时 queue/history 缺席推断终态。
同一 endpoint 的新 dispatcher 还必须等待所有已绑定、由 recovery 持有的旧 child（包括 Standard）获得
精确定向取消或终态证明，防止旧 `/prompt` 迟到入队后越过新的 Ray/Standard 切换顺序。
Director 会在 POST `/prompt` 前保存 caller-assigned prompt ID 并先连接 WebSocket；即使首个 `executing`
早于 POST 响应，也允许精确匹配的 `preparing/submitting` child 直接进入 running，随后提交确认不得把阶段
覆盖回 queued。首次连接只做至多 1 秒的有界握手等待；超时继续提交并由 queue/history 兜底。错误 endpoint、
缺 prompt ID、未知节点以及任何终态 child 的迟到事件一律忽略。

WebSocket 的二进制预览只接受带 metadata 的 event-4。后端严格解析长度与 JSON，只接收登记 child 的
sampler node 所属 prompt，且格式必须为 PNG/JPEG、大小不超过 2 MiB。最新帧只存在有 TTL 和总量上限的
内存缓存；`/api/jobs/{job_id}/live-preview` 仅在该 child 仍 queued/running 时返回，带 `no-store` 与
`nosniff`，任务删除或 child 终态会使其失效。服务重启会为历史活动 endpoint 恢复标准 WS monitor，
但不会持久化旧预览帧。

每个成功 child 只能通过自身 `output_nodes[segment_id]` 声明的 SaveVideo node 归属输出。
`segment_results[]` 只暴露恰好一份非歧义映射：稳定 `segment_id`、`child_id`、受控输出 URL、文件名
和 `current_snapshot`，不泄露原始 prompt。`current_snapshot` 由服务端对完整 timeline 与 runtime
settings 权威快照做严格相等判断；历史映射不会被删除，但不能进入主监视器。完整任务在 assembly 前
要求期望 segment 集合与输出集合完全相等；缺失、重复或未知映射都会失败，不能用旧缓存、源视频或
占位补洞。一个 child 失败时，其余 child 继续对账并保留已经生成的候选 take。
