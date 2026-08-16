# 两种编辑族与六种原生配方契约

本文档描述统一长视频时间线 v2 的当前服务端契约。编辑器只选择 MiniMax H3 的两个模型族
`fl2va | ref2va`；后端在 compile/submit 时根据 typed 素材确定 T2V、I2V、FL2V、R2V、V2V、
RV2V 六种原生配方之一。它不是 ComfyUI workflow 格式；浏览器只能提交 typed timeline 和资产
ID，后端始终从固定原生模板构造 API prompt。

## 共享字段

每个分段都包含：

- `id`：时间线内稳定且唯一的标识；
- `title`、`prompt`、`duration_seconds`、`enabled`；
- `mode`：`fl2va | ref2va`，表示用户选择的模型族，不表示执行配方。

时间线 wire `version` 为 `2`，共享 `render`、按模型族拆分的 `sampling.fl2va` /
`sampling.ref2va`、`ref_image_size`、`audio_mode`、`export_mode` 和 `continuity`。原生执行模板
v1 额外定义：

- `render.fps == 24`；
- 不保存或展示 CFG；MiniMax H3 原生 `BasicGuider` 没有该输入；
- `continuity.overlap_frames` 只允许 H3 有效 guide 长度 `5 | 22 | 39 | 56`；
- 接续开启时，首个启用段和带显式 `first_image` 的 FL2VA 段是锚点重置；其余所选段依赖完整启用
  时间线中紧邻的前一个启用段，前后段可跨 `mode`/recipe/family。该前驱未在本次选择中时，服务端必须
  解析当前 endpoint 上同稳定 segment ID 且输出几何一致的最新成功历史 take；浏览器不能指定 take 或路径；
- 最多 128 个分段；
- 不接受负面提示词字段。

模型族切换必须重建判别联合对象。另一模型族的专属字段属于非法额外字段，不能隐藏保留。

## 两种可编辑形状

- `mode=fl2va`：统一包含可空 `first_image` 与 `last_image` 两个图片锚点。
- `mode=ref2va`：统一包含可空 `source_video`、`source_start_seconds`、
  `source_duration_seconds`、`source_audio_as_reference`，以及局部分槽的
  `reference_images`、`reference_audios`、`reference_videos`。

`source_video` 是要编辑/延续的主视频，`reference_videos` 是独立参考网格；两者不能合并成一个
含糊数组。源视频的 `source_start_seconds + source_duration_seconds` 不能超过服务端登记的代理
时长。视频上传时统一转为不可变 24fps 代理，因此 workflow 不接受客户端声称的帧率或路径。

Ref2VA 可以同时使用源视频与独立参考视频。官方 `MiniMaxH3ReferenceToVideo` 最多接三路参考
视频，因此有源视频时它占第 0 路，独立 `reference_videos` 最多两路；无源视频时最多三路。

## 服务端确定性配方推导

| `mode` | typed 素材形状 | `recipe` | 原生 conditioning |
| --- | --- | --- | --- |
| FL2VA | 无首尾锚点 | T2V | `MiniMaxH3ImageToVideo` 无帧输入 |
| FL2VA | 只有 `first_image` | I2V | `first_frame` |
| FL2VA | 存在 `last_image`（尾图单独或首尾都有） | FL2V | `last_frame`，可同时有 `first_frame` |
| Ref2VA | 无源视频，至少一种独立参考 | R2V | 独立图片/音频/视频参考 |
| Ref2VA | 有源视频，无独立参考 | V2V | 源视频作为第一个视频参考 |
| Ref2VA | 有源视频，且有任意独立图片/音频/视频参考 | RV2V | 源视频加独立参考 |

空 Ref2VA 分段是合法的未完成编辑状态，但 compile/submit 必须失败封闭；不能把它偷偷改成 FL2VA
或发出无参考的 Ref2VA 图。脱敏 compile plan 同时返回 `mode=fl2va|ref2va` 与六值 `recipe`，不能
复用 `mode` 字段偷换含义。

## 参考槽与提示词标签

FL2VA 的图片标签不是持久化 slot，而是由实际送入
`MiniMaxH3ImageToVideo` 的锚点顺序派生：仅首图或仅尾图时，该图都是
`<Picture 1>`；首尾图同时存在时，首图是 `<Picture 1>`、尾图是
`<Picture 2>`。增删锚点导致序号变化时，必须按素材身份同步重写提示词，不能让旧标签静默指向另一张图。

Ref2VA 的 `reference_images`、`reference_audios`、`reference_videos` 各自使用局部分槽。
每种 modality 的 slot 必须密集排列为 `0..N-1`；提示词展示标签分别为
`<Picture 1..N>`、`<Audio 1..N>`、`<Video 1..N>`。后端不为稀疏槽偷偷重编号。

主视频固定占提示词语义上的 `<Video 1>`，独立参考视频从 `<Video 2>` 开始；无主视频时独立
参考视频从 `<Video 1>` 开始。Ref2VA 可用分段专属
`source_audio_as_reference` 把经服务端确认存在的主视频音轨同时作为配对 `<Audio 1>` 条件；
此时独立 `reference_audios` 的内部 slot 仍是 `0..N-1`，提示标签整体变为
`<Audio 2..N+1>`。静音视频以及缺少 `has_audio` 的历史 metadata 均失败封闭。R2V 参考视频不自动
配对音轨。

提示词出现的标签必须在当前分段存在。标签不会跨分段引用，也不等于资产库的网格序号。带前驱接续的
仅尾图 FL2VA 段仍把尾图送入 `MiniMaxH3ImageToVideo` 建立 `<Picture 1>` 视觉块，并用同一张图额外
guide 到最终可见帧；alignment tail 上的同图隐式锚点会随隐藏尾部一起裁掉。

## 素材身份

素材对象中的 `id` 是唯一可信入口。提交和保存时，后端会用 SQLite 中不可变登记记录校验：

- `name`、`subfolder`、`type`、`kind`；
- canonical ComfyUI origin；
- 视频 probe metadata；
- 服务端内容哈希。

客户端路径不能覆盖这些字段。切换 ComfyUI endpoint 后，旧 endpoint 的资产不能用于新任务。
删除仍被统一时间线或旧草稿引用的资产记录返回 409；删除记录不会默认删除远端文件。
显式 `cascade=true` 才会在一个 immediate transaction 内解除所有 typed 引用、压密剩余 slot 并
同步重写该分段提示词标签。删除素材不会替用户跨模型族：FL2VA/Ref2VA `mode` 保持不变，下一次
compile 由剩余素材重新推导 recipe。Ref2VA 失去最后一个有效源/参考后可继续作为未完成编辑保存，
但不能提交；若源音频策略不再有效则归一为 `generate`。任一文档重新校验失败时，资产和所有草稿
保持原样。

## v1 兼容迁移

服务端继续接受并严格验证旧时间线六模式判别联合，再无损归一为 v2：

- `t2v | i2v | fl2v` → `mode=fl2va`，保留已有首尾锚点；
- `r2v | v2v | rv2v` → `mode=ref2va`，保留源视频、裁剪、源音轨开关及所有旧参考字段；
- 旧非法跨模式字段仍然报错，不会因为 v2 形状更宽就意外生效；
- 旧 `version=1` 活草稿在数据库初始化时规范化为 `version=2`，历史 job 快照和旧六模式草稿接口
  继续可读，不改写执行审计。

## 执行与导出

后端为每个实际所选启用段构造一个 server-owned workflow unit。同族同后端 unit 的 loader
节点 ID/输入保持稳定，由 ComfyUI 跨 prompt 复用；每段拥有独立 conditioning、sampler、decode、
`SaveVideo` 和失败/取消边界。关闭接续时，提交顺序固定为 Standard→RayLight、FL2VA→Ref2VA、
时间线顺序；开启接续时保持时间线顺序，同次运行的依赖边必须等待前驱成功后再提交；已在预检前绑定的
历史 take 边不等待旧任务。不能按后端或模型族重排，多个 RayLight unit 始终串行。

`segment_ids` 只校验和生成所选段，未完成的未选段不会阻塞生成所选。开启接续后可以单独选择没有显式
首图重置的后段，但其当前直系前驱必须存在同 endpoint、同稳定 segment ID、同宽高/FPS/可见帧数的成功历史
take；没有 take、输出几何不兼容或生成音频所需 take 不含音轨时以精确 422 失败。完整生成要求每个启用段
恰有一个与本次 child output-node 映射一致的结果；缺失、重复和未知结果都使父任务失败。
完整 `export_mode=all` 使用 ffmpeg 按时间线顺序统一规格并拼接；选择生成只返回片段结果。
任务响应的 `segment_results[]` 按稳定 segment ID 暴露非歧义候选及受控输出 URL，并携带
`current_snapshot`。该值只在任务保存的完整 timeline 与 runtime settings 快照都严格等于当前
服务器权威值时为真；旧 take 仍可在任务抽屉查看，但不会混入主监视器。单段失败不会删除其他
成功 child 的候选。浏览器不会收到 child 的原始 API prompt。

成功分段另登记在无任务外键的 `segment_takes` 账本中，删除任务记录不会删除 take。匹配键由稳定
segment ID 与服务端输出几何指纹组成；指纹只含项目宽高、FPS 与 H3 对齐后的实际可见帧数。提示词、
`mode`/recipe/family、首尾锚点、源视频/裁剪、参考素材与 slot、标题、启用状态、顺序、接续、导出、
模型、LoRA、采样、Seed、GPU 和其他 runtime 设置都不参与。查询还严格绑定当前 canonical ComfyUI
origin；`audio_mode=generate` 额外要求 take 的成片具备音轨，但音频策略本身不进入几何指纹。
计划只暴露 `continuity_source=same_run|historical_take` 和 opaque `historical_take_id`，不暴露输出路径。

`audio_mode=source` 仅允许所选段全部为带 `source_video` 的 Ref2VA、源视频经服务端确认包含音轨，并要求源裁剪帧数
和输出对齐帧数相同，首版不隐式拉伸源音频。它只决定最终 `CreateVideo` 的音轨，和上述 H3 条件
开关彼此独立。

接续段设原可见输出帧数为 F、接续帧数为 N。后端从本次前驱成功 child 声明的唯一 `SaveVideo` 输出，
或服务端解析并在 runtime 预检前绑定的历史 take 读取
最后 N 帧，以官方 `MiniMaxH3AddGuide(frame_idx=0)` 作为后段开头 guide，并采样
`align(F+N)` 帧（向上吸附到下一个 `17k+5`）；保存前只截取 `[N, N+F)`，同时丢弃对齐产生的
尾帧，因此启用接续前后分段输出和全片总帧数不变。`audio_mode=generate` 同时接入前驱尾部音频并把
生成音频裁到同一可见区间；`source`
继续使用当前段自己的源音轨，`mute` 不写音轨。依赖 FL2VA 段若有 `last_image`，它锚定在
`N+F-1`（最后一个可见采样帧），不能落到随后会被裁掉的 alignment tail。

## 采样、LoRA 与设备

FL2VA 与 Ref2VA 分别保存步数、Seed、采样器、调度器和 Video/Audio Shift。固定或随机 Seed 都必须
不大于 `Number.MAX_SAFE_INTEGER`，以保证 JSON 经浏览器编辑后仍完全相同。勾选随机时，浏览器在
每次正式提交前重掷一个安全整数，并先把它显示到灰显数字框；服务端不再生成隐藏随机值。compile
报告用 `seed_mode=random`（或 `fixed`）并始终返回该图携带的确切 `seed`。LoRA 位于模型族
共享配置中只选择 LoRA 文件与强度。后端按经过审计的精确 basename 自动插入 H3 Turbo 专用、
量化旁路或通用 model-only loader；RayLight 固定插入 `RayLoraLoader`。未知或相似伪装命名失败封闭，
旧 `lora_loader` 不再影响图。不选 LoRA 不要求任何 LoRA 自定义节点。

单卡 GPU 池自动使用 Standard、模型绑定的 `device` 和官方 `Select*Device` 节点；两卡或以上自动使用
RayLight、`gpu_select` 与并行拓扑。所有 logical GPU 在提交前与 `/system_stats` 对照，缺失时 409，不能
让 selector 静默回到默认卡。RayLight 拓扑乘积必须等于 GPU 数。
原生时间线 v1 将 `fsdp` 与 `cpu_offload` 固定为 `false`，直到 RayLight 的采样后 CUDA
清理完成 U2/U4 实机验证。

完整节点白名单、父子任务和进度契约见 [native-workflow-execution.md](native-workflow-execution.md)。
