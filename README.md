# Director Web

面向 MiniMax H3 长视频创作的本地网页版导演台。工作对象是一条统一时间线；每个分段
只选择 FL2VA 或 Ref2VA 模型族，后端再按绑定素材推导 T2V、I2V、FL2V、R2V、V2V 或
RV2V 执行配方，而不是先进入某一种生成模式。

Director 是纯 ComfyUI 插件（ComfyUI Registry 发布名 **DirectorDeck**）：后端嵌入
ComfyUI 进程运行，前端页面由 ComfyUI 托管在 `/directordeck/`，数据库固定在 ComfyUI
`user/directordeck/database/`。不再有独立部署形态。

## 执行边界

- 浏览器只提交严格校验的时间线、素材 ID、分段选择和系统设置，不能提交 ComfyUI
  workflow、API prompt、`class_type` 或节点连线。
- 后端从固定、版本化模板构造 API prompt，并在每次提交前重新预检。
- 标准模板只使用 ComfyUI core 与官方 `comfy_extras` 节点；不使用
  `MiniMaxH3Director` 自定义节点，也不依赖其上传、探测、分镜或进度接口。
- Standard LoRA 加载节点由用户安装并通过精确配置映射选择；该映射是用户权威，Director
  不按模块名、接口切片或实现指纹拒绝外部或用户修改过的节点。真实导入、提示词校验和执行错误
  在发生时按原始错误报告。多卡执行只使用 DirectorDeck 自带并维护的 RayLight 分支；该分支以
  `DirectorDeckRay*` 专属节点名注册，并运行在私有 `directordeck_raylight` Python 命名空间；
  外部 `custom_nodes/raylight` 不会被 Director 导入或选用。

一次时间线提交产生一个父任务，并为每个所选分段建立一个独立原生 child prompt。各 prompt
使用稳定的模型、CLIP、VAE loader 节点 ID/输入，让 ComfyUI 可以跨 prompt 复用缓存，同时把
失败、取消和解码中间值限制在单段。所有预期分段成功后，后端按稳定 segment ID 校验输出并用 ffmpeg 组装长片；
缺少、重复或额外输出都不会被误报为成功。

详细节点链、白名单和选择规则见
[原生工作流执行架构](docs/native-workflow-execution.md)，统一时间线数据模型见
[长视频时间线设计](docs/unified-timeline-design.md)。

## 当前能力

- React 统一长视频工作区：资产库、主预览、操作栏、主时间线和分段检查器；
- 每段只选择 FL2VA/Ref2VA 模型族；后端按首尾锚点、源视频和独立参考素材确定性推导六种 H3
  原生配方，compile 计划分别报告 `mode` 与 `recipe`；
- FL2VA 首尾锚点按实际输入顺序派生 `<Picture N>`；Ref2VA 图片、视频、音频参考使用分段局部稳定
  slot。两类标签都在界面与服务端校验；Ref2VA 按 H3 原生能力统一限制为 9 张参考图片、3 路视频和
  3 条独立参考音频，源视频占用其中 1 路视频；
- FastAPI + SQLite 持久化统一时间线、共享设置、资产、父任务和逐段 child 执行单元；
- 视频上传后由后端 ffmpeg/ffprobe 生成并登记不可变 24fps 代理；智能分镜也在后端执行；
- FL2VA 与 Ref2VA 是独立模型配置槽，但都可选择 ComfyUI 返回的完整 diffusion 清单；
- 每个模型族配置一个逻辑 GPU 池：单卡自动使用 Standard，两张或以上自动使用 RayLight；
- 派生出的 RayLight 缺节点、GPU 或拓扑错误时直接阻断，不会静默退回 Standard；RayLight 默认按
  完整 RayLight 配置 key 保留 worker 权重，同 key 复用，不兼容 key 或 Standard 会先安全释放旧池；
- UNET、CLIP、Video VAE、Audio VAE 可配置 ComfyUI 逻辑设备；`gpu:N` 不是物理卡号；
- LoRA 只选择文件和强度；Standard 加载器采用用户配置的精确映射，Director 不维护或鉴定其
  第三方实现，RayLight 链使用随插件维护的 `DirectorDeckRayLoraLoader`。真实缺失、导入和执行错误按原始错误报告；
- MiniMax H3 原生模板固定 24fps 并使用 `BasicGuider`；产品不提供没有实际作用的 CFG 或负面提示词；
- FL2VA 与 Ref2VA 各自保存步数、Seed、采样器、调度器和 Video/Audio Shift。Seed 始终是浏览器可
  无损往返的 JavaScript safe integer；勾选随机时，前端在每次提交前重掷并把确切数值显示在灰显输入框，
  编译报告、任务快照与 ComfyUI prompt 使用同一个值；
- ComfyUI 标准 WebSocket 的 `execution_start/executing` 显示模型加载、条件构建、采样、解码和保存阶段；
  能向主进程上报 `progress` 的 sampler 另外提供逐段 child 的精确 `step/total`。当前 RayLight worker
  通常只保证采样阶段可见，不承诺内部逐步数字；每个父任务每轮只读取一次 queue，
  再对离队 child 使用一次受限 bulk history。queue 可用时，遗漏项按轮转窗口最多精确补查 16 项；
  queue 不可用时只接受 bulk 中的确定完成项，不逐段扇出请求，也不凭“缺席”误判取消；
- metadata-bearing event-4 实时预览仅在内存中短期缓存，并严格校验
  prompt、sampler、图片类型和 2 MiB 大小上限；公开 URL 仅在对应 child 仍活动且缓存命中时出现，
  响应禁止缓存和 MIME 嗅探；
- 安全取消依赖当前 ComfyUI 原生原子 job-cancel API；旧服务器缺少该能力时界面禁用取消，
  不会退回可能误杀其他任务的全局 interrupt；
- 终态任务支持单条删除和批量清理。删除导演台记录不会越权删除 ComfyUI 文件；
- 每个非歧义分段输出通过稳定 `segment_results` 映射回 timeline；只有任务的完整 timeline 和
  runtime settings 快照都与当前服务器权威值严格相同时，主监视器才显示该候选。本地尚未完成自动同步的编辑
  期间也会隐藏候选；历史 take 仍保留在任务抽屉中；完整长片仍由父任务单独暴露组装后的 output；
- 被草稿引用的资产默认拒绝移除；显式级联只在一个事务中解除 typed 引用、修复 slot/提示词并
  保持模型族、按剩余素材重算配方，任一步失败整笔回滚；它不删除 ComfyUI 输入或任何生成输出；
- 深色和 Claude 风暖色浅色主题；左右栏及全局设置浮层不会挤占时间线工作区。

原生时间线现在支持基于官方 `MiniMaxH3AddGuide` 的逐段接续。启用 `continuity` 后，首个启用段以及
带显式 `first_image` 的 FL2VA 段是锚点重置；其余段依赖时间线上紧邻的前一个启用段，T2V、FL2V 与
Ref2V 可以混排。前驱既可以在同一次运行中生成，也可以复用当前 ComfyUI 实例上该稳定分段 ID 的最新
成功成片；历史成片只匹配宽高、FPS 和 H3 可见帧数，不因提示词、生成配方、参考素材、模型、LoRA 或
推理参数变化而失效。后端确认前驱的唯一 `SaveVideo` 输出后，才把该 `[output]` 动态绑定到后段的
`LoadVideo` 并提交。后段取前驱最后 N 帧（N 为
5/22/39/56）在 `frame_idx=0` 作为官方 H3 guide，采样 `align(F+N)` 帧，保存前裁掉前置 N 帧和对齐产生的
尾帧，因此每段仍只输出原来的可见 F 帧，接续不会增加全片时长或在拼接处重复尾帧。生成音频随同接续并按
相同可见区间裁剪；保留源音频与静音策略不把前驱音轨写入输出。这是显式尾帧条件接续，不宣称等同于旧
Director 的跨段 AV latent handoff，也不是剪辑层 crossfade。前驱失败、被 ComfyUI 外部取消、输出缺失或
歧义时，尚未提交的依赖后段以 `status=failed, stage=dependency_failed` 收口，不会带着未绑定占位路径
进入 ComfyUI；用户取消整个父任务则立即停止续提，未提交后段按取消语义收口。

纯 Standard 任务不主动执行段间清显存，稳定 loader 输入有利于跨任务复用。RayLight 默认使用
`keep_until_switch`：family、model、LoRA、GPU pool、topology 与会修改 worker 的 sigma shift 完全相同的后续分段和任务直接复用 CUDA
权重；任一项变化时，Director 会在提交锁内先提交并等待 `DirectorDeckRayKill` 安全屏障，再用递增 epoch
创建新池。切到 Standard 也先清旧 Ray 池，因此 FL2VA 与 Ref2VA 可以在同一组 GPU 上顺序运行，无需重启
ComfyUI，也不依赖 OOM 自动卸载。Director 还固定发送 `driver_cleanup_policy=ray_devices`：采样前只释放
`GPU_SELECT` 所列 Ray 逻辑卡上的 Comfy driver 模型，不会强制卸载放在非 Ray 卡上的 CLIP/VAE；这些非 Ray
卡缓存仍由 ComfyUI 自动管理，可能在显存或内存压力、切换模型、Free Memory 或重启时被淘汰，并非绝对常驻。
明确选择“任务后释放”则每次采样把 worker 模型从 CUDA 卸载，采样后的 driver 清理也沿用同一
`ray_devices` 范围；辅助模型与 Ray 共用卡时，这会先为后续 VAE 解码或其他负载安全腾出该卡。安装版
RayLight 会保留 ModelPatcher 与 actor handle，因此同 key 可以沿用 epoch/loader 缓存，并在下次 sampler
开始时把权重重新载入 CUDA。Ray cluster 仍会被追踪，切到 Standard 前同样先完整 shutdown。
Director 每次只向 ComfyUI 提交一个 Ray 生成段，必须等该 prompt 的 exact history 到达成功终态，才提交
同一父任务或后一父任务的下一段；失败或外部移除会先 taint 旧池并执行 RayKill，再用新 epoch 继续剩余段。
创建接口在预检成功且任务/分段已持久化、提交顺序票据已登记后立即返回 `preparing`，不会占住浏览器请求。
这些保证只覆盖经过 Director 提交锁排队的任务；在 ComfyUI Web 手工提交的 workflow 不受
Director 调度，可能使持久状态失真且无法保证被自动检测，不能把手工混跑当成受支持的并发路径。
`keep_until_switch` 只承诺兼容 Ray worker 池的生命周期；若 CLIP/VAE 与该池共用同一张卡，跨进程显存
无法由 ComfyUI 单方面自动协调，不能同时承诺辅助模型常驻，显存不足时应改用“任务后释放”或分离设备。

## 生成文件与任务记录

每个分段由 `SaveVideo` 写入任务设置快照对应的 ComfyUI output，最终长片上传到该实例的
`output/directordeck/timelines/`。SQLite 只保存 `filename/subfolder/type` 引用和审计快照，
不复制媒体正文。因此：

- 在 ComfyUI 界面清 history 不等于删除磁盘文件；
- 在 ComfyUI 删除文件不会自动删除导演台任务，预览会随文件消失而失效；
- 在导演台删除任务只删除本地记录，输出文件仍由 ComfyUI 管理。

素材正文保存在 ComfyUI 的 input 目录，数据库只记录相对路径引用；二者都绑定当前
ComfyUI 安装，迁移到另一套 ComfyUI 时需要重新上传素材。

## 安装

Director 以 **DirectorDeck** 为名发布到 ComfyUI Registry。精确兼容基线见
[发布说明](RELEASE.md)。模型权重、LoRA 和用户素材不随插件分发。

- **ComfyUI Manager（推荐）**：在 Manager 中搜索 `DirectorDeck` 安装，然后重启 ComfyUI。
- **手动安装**：把插件仓库克隆到 ComfyUI 的 `custom_nodes/` 下，并用 ComfyUI 的
  Python 环境安装依赖，然后重启 ComfyUI：

  ```bash
  cd /path/to/ComfyUI/custom_nodes
  git clone https://github.com/JYE-HC/DirectorDeck.git
  /path/to/ComfyUI/.venv/bin/pip install -r DirectorDeck/requirements.txt
  ```

  Windows portable 版改用其自带的 `python_embeded` 解释器执行同一 pip 命令。

从本开发仓库联调时，改为构建插件包并软链进 ComfyUI：

```bash
cd frontend && npm ci && npm run build && cd ..
python3 tools/build_plugin.py --link /path/to/ComfyUI
```

## 使用

重启 ComfyUI 后，通过顶部菜单或侧栏的 **Director** 入口打开导演台；浏览器直接访问
`http://<ComfyUI 地址>/directordeck/` 亦可。后端嵌入 ComfyUI 进程，ComfyUI 连接地址由插件自动
推导注入，无需手动配置。

Director 没有登录鉴权，随 ComfyUI 同源提供服务。不要把 ComfyUI 直接暴露到公网；
跨机器访问前应在反向代理层增加 TLS、身份认证与来源限制。

## 多卡（RayLight）

单卡自动走 Standard 链，只使用 ComfyUI core 与官方 extras。两张及以上 GPU 时在“系统设置”
开启多卡推理，并按提示在插件内安装 RayLight 依赖（`requirements-raylight.txt`）后重启
ComfyUI。多卡仅支持 Linux。Standard LoRA 的加载节点由用户在 ComfyUI 中自行安装并通过精确
映射选择；DirectorDeck 不打包或维护第三方 LoRA 节点。Director 不依赖、也不会生成旧
`MiniMaxH3Director` 大节点；它是否安装不会影响 DirectorDeck 使用。

## ffmpeg

上传探测与长片组装需要 ffmpeg/ffprobe。系统缺失时可在“系统设置”的媒体工具面板一键安装
static-ffmpeg，无需手动配置环境。

## 数据位置

数据库固定为 ComfyUI `user/directordeck/database/directordeck.sqlite3`，随 ComfyUI 安装走，
不在插件目录内。独立部署形态已移除，不提供旧数据的迁移路径。

`directordeck.sqlite3-wal` 与 `directordeck.sqlite3-shm` 是活动数据库的事务旁路文件，
`directordeck.sqlite3.instance.lock` 是单实例锁元数据；它们都不是额外工作区，也不应在运行时单独
复制或删除。手工备份文件只有在明确恢复时才会使用，Director 不会自动把名字相似的备份当作
当前数据库。

## 开发与完整验证

```bash
uv sync --all-groups
uv run pytest backend/tests -q
cd frontend
npm ci
npm test
npm run build
```

轻量验证不需要模型或素材。完整原生 prompt 验证需要准备一个图片、一个 24fps 视频、旧 Turbo
LoRA 和一个已有输出视频，并以 ComfyUI 相对路径传入：

```bash
export COMFYUI_ROOT=/path/to/ComfyUI
export NATIVE_VALIDATION_IMAGE=director-validation/image.png
export NATIVE_VALIDATION_VIDEO=director-validation/video.mp4
export NATIVE_VALIDATION_OUTPUT_VIDEO=director-validation/output.mp4
PYTHONPATH=backend "$COMFYUI_ROOT/.venv/bin/python" \
  tools/validate_native_comfy_prompts.py
```

该脚本仅做 CPU 注册表和 prompt 结构验证，不加载模型或采样。发布前仍需分别执行一次真实 Standard
和所选 RayLight GPU 拓扑的生成冒烟测试。

## 许可证

Director Web 采用 [GNU General Public License v3.0](LICENSE)，SPDX 标识为
`GPL-3.0-only`。仓库内第三方组件继续适用各自保留的许可证，详见
[第三方声明](THIRD_PARTY_NOTICES.md)。
