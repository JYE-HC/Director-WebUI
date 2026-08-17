# Director Web

面向 MiniMax H3 长视频创作的本地网页版导演台。工作对象是一条统一时间线；每个分段
只选择 FL2VA 或 Ref2VA 模型族，后端再按绑定素材推导 T2V、I2V、FL2V、R2V、V2V 或
RV2V 执行配方，而不是先进入某一种生成模式。

## 执行边界

- 浏览器只提交严格校验的时间线、素材 ID、分段选择和系统设置，不能提交 ComfyUI
  workflow、API prompt、`class_type` 或节点连线。
- 后端从固定、版本化模板构造 API prompt，并在每次提交前重新预检。
- 标准模板只使用 ComfyUI core 与官方 `comfy_extras` 节点；不使用
  `MiniMaxH3Director` 自定义节点，也不依赖其上传、探测、分镜或进度接口。
- 自定义节点只允许两类明确例外：后端从受支持 H3 LoRA 文件名推导出的加载器，以及由
  多卡 GPU 池自动启用的 RayLight 执行链。两类例外都有精确白名单，缺节点或配置不兼容时直接失败。

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
  旧 `backend` 设置只为兼容升级而接受，保存后统一归一为 `auto`，不能隐藏覆盖 GPU 池；
- 派生出的 RayLight 缺节点、GPU 或拓扑错误时直接阻断，不会静默退回 Standard；RayLight 默认按
  完整 RayLight 配置 key 保留 worker 权重，同 key 复用，不兼容 key 或 Standard 会先安全释放旧池；
- UNET、CLIP、Video VAE、Audio VAE 可配置 ComfyUI 逻辑设备；`gpu:N` 不是物理卡号；
- LoRA 只选择文件和强度，加载器由后端自动确定：旧 H3 Turbo 使用专用节点，当前量化兼容版使用
  旁路节点，当前 `comfyui_bf16` 版使用通用 model-only，RayLight 使用 `RayLoraLoader`；未知命名失败封闭；
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
- 终态任务支持单条删除和批量清理。删除导演台记录不会越权删除远程 ComfyUI 文件；
- 每个非歧义分段输出通过稳定 `segment_results` 映射回 timeline；只有任务的完整 timeline 和
  runtime settings 快照都与当前服务器权威值严格相同时，主监视器才显示该候选。本地尚未完成自动同步的编辑
  期间也会隐藏候选；历史 take 仍保留在任务抽屉中；完整长片仍由父任务单独暴露组装后的 output；
- 被草稿引用的资产默认拒绝移除；显式级联只在一个事务中解除 typed 引用、修复 slot/提示词并
  保持模型族、按剩余素材重算配方，任一步失败整笔回滚；它不删除 ComfyUI 远程输入或任何生成输出；
- 深色和 Claude 风暖色浅色主题；左右栏及全局设置浮层不会挤占时间线工作区。

原生时间线现在支持基于官方 `MiniMaxH3AddGuide` 的逐段接续。启用 `continuity` 后，首个启用段以及
带显式 `first_image` 的 FL2VA 段是锚点重置；其余段依赖时间线上紧邻的前一个启用段，T2V、FL2V 与
Ref2V 可以混排。前驱既可以在同一次运行中生成，也可以复用当前 ComfyUI 端点上该稳定分段 ID 的最新
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
权重；任一项变化时，Director 会在 endpoint 提交锁内先提交并等待 `RayKill` 安全屏障，再用递增 epoch
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
创建接口在预检成功且任务/分段已持久化、endpoint 顺序票据已登记后立即返回 `preparing`，不会占住浏览器请求。
这些保证只覆盖经过 Director endpoint 提交锁排队的任务；在 ComfyUI Web 手工提交的 workflow 不受
Director 调度，可能使持久状态失真且无法保证被自动检测，不能把手工混跑当成受支持的并发路径。
`keep_until_switch` 只承诺兼容 Ray worker 池的生命周期；若 CLIP/VAE 与该池共用同一张卡，跨进程显存
无法由 ComfyUI 单方面自动协调，不能同时承诺辅助模型常驻，显存不足时应改用“任务后释放”或分离设备。

极少数预发布版本写入的 v1 RayLight 账本缺少可验证的完整 loader chain。升级后 Director 会保留其中的
epoch，但把可能仍存活的旧 actor 标成未知：在一次 Director RayLight 任务用新 epoch 显式重建运行池前，
Standard 提交会失败封闭，不能把“旧账本无法描述”误当成“显存里一定没有旧池”。

## 生成文件与任务记录

每个分段由 `SaveVideo` 写入任务设置快照对应的 ComfyUI output，最终长片上传到该实例的
`output/director-web/timelines/`。SQLite 只保存 `filename/subfolder/type` 引用和审计快照，
不复制媒体正文。因此：

- 在 ComfyUI 界面清 history 不等于删除磁盘文件；
- 在 ComfyUI 删除文件不会自动删除导演台任务，预览会随文件消失而失效；
- 在导演台删除任务只删除本地记录，输出文件仍由 ComfyUI 管理。

资产同时绑定上传时的 canonical ComfyUI origin。切换服务器后必须在新实例重新上传，避免
相同相对路径被误认为同一份素材。

## 安装与启动

这是 Linux 本地部署的首个 alpha 发布候选。精确兼容基线和依赖来源见
[发布说明](RELEASE.md)。模型权重、LoRA 和用户素材不随本仓库分发。

推荐使用仓库根目录的一键引导脚本，依次完成平台检测、系统依赖、uv、Node.js、ffmpeg、
ComfyUI 准备、Director 安装与验证。先运行只读检查，再安装：

```bash
./bootstrap.sh check     # 只读环境检查，不写入任何文件
./bootstrap.sh install   # 交互式安装；加 -y 使用推荐默认值
./bootstrap.sh verify    # 只验证已安装内容
```

安装按固定步骤队列执行：检测平台、检查项目文件系统位置、选择 ComfyUI 安装方式、安装系统
依赖、uv、Node.js、ffmpeg、准备 ComfyUI、安装 ComfyUI 依赖、安装 Director 与 custom
nodes、离线验证、可选在线验证、生成启动配置、可选启动服务。`--dry-run` 只打印检测结果与
安装计划；`--resume`/`--from`/`--only`/`--skip` 控制断点续装与步骤裁剪；`--no-sudo`
禁止安装系统包，uv/node/ffmpeg 尽量装入项目 `.tools/`；已有兼容 Node.js 时可用
`--node-bin-dir PATH` 直接复用。

ComfyUI 有三种接入方式：自动 clone（默认，`--comfyui-ref latest|tested|<tag/sha>`）、复用
已有安装（`--comfyui-root PATH`，可用 `--comfy-python PATH` 指定解释器）、跳过本地安装直连
远程实例（`--skip-comfyui --comfyui-url URL`）。WSL2 下项目位于 /mnt/c 时会自动迁移到
WSL 原生文件系统（`--skip-relocation` 关闭），并可用 `--windows-comfyui-root` 复用
Windows 侧 ComfyUI 的模型目录。

本地 ComfyUI 模式下，Director 前后端与 bundled custom nodes 的安装及离线验证由发布包内的
`install.sh` 完成。安装器会校验本发布包、ComfyUI Git 能力、Python/CUDA/Ray 环境、Node、ffmpeg
和 custom node 冲突；它不会执行 `git pull`、切换 ComfyUI commit、修补 ComfyUI 核心、下载模型
或自动重启 ComfyUI。与实测 ComfyUI commit 相同或属于它的后继版本均可安装；本地改动只提示
统计信息，不会展示文件名，也不会阻止安装。若已有同名但内容不同的节点，安装会停止；只有在
ComfyUI 已停止时，才可显式使用 `--replace-node raylight --replace-node ComfyUI-MiniMax-H3-Turbo
--confirm-comfyui-stopped`。旧目录会保存在 ComfyUI 根下的 `.director-backups/`，不会被删除。

标准单卡链只使用 ComfyUI core/官方 extras。两张及以上 GPU 才会使用仓库内的 Director 定制
RayLight；`ComfyUI-MiniMax-H3-Turbo` 只在选择旧版专用 Turbo LoRA 时使用。不要安装旧
`MiniMaxH3Director` 大节点，Director 不依赖并明确拒绝它。

安装与服务管理共用同一入口。`start`/`stop`/`restart`/`status`/`logs` 同时管理 Director 与本地
ComfyUI（远程 ComfyUI 模式下只管理 Director）；带 `-director` 或 `-comfyui` 后缀的命令只管理
对应一方：

```bash
./bootstrap.sh start|stop|restart|status|logs   # Director + 本地 ComfyUI
./bootstrap.sh start-director|stop-director|restart-director|status-director|logs-director
./bootstrap.sh start-comfyui|stop-comfyui|restart-comfyui|status-comfyui|logs-comfyui
./bootstrap.sh reset    # 只清空 .director-install 安装状态，不删除 ComfyUI、数据库或模型
```

`logs` 不带参数时同时跟踪 backend/frontend/comfyui 三份日志，也可用
`logs backend|frontend|comfyui` 只看一份。

监听地址和端口可在安装或 `start`/`restart` 时用 `--listen-host`、`--backend-port`
（默认 8787）、`--frontend-port`（默认 4173）覆盖；`--start`/`--start-comfyui` 让安装
成功后直接拉起对应服务。例如让前后端监听所有 IPv4 网卡：

```bash
./bootstrap.sh restart --host 0.0.0.0 --backend-port 8788 --frontend-port 4174
```

`director.sh` 使用用户级临时 systemd unit 管理前后端进程，默认只监听 `127.0.0.1`，日志写入
`data/director-backend.log` 和 `data/director-frontend.log`，可用 `./bootstrap.sh logs backend`
或 `./bootstrap.sh logs frontend` 跟踪。`0.0.0.0` 仅用于监听，浏览器访问时应使用服务器实际 IP。

不具备引导脚本条件时也可手动安装。后端：

```bash
uv sync --all-groups
uv run director-web
```

前端：

```bash
cd frontend
npm ci        # 生产构建：npm run build；开发调试：npm run dev
```

Vite 仅监听本机并把 `/api` 代理到后端。首次启动没有写死的 ComfyUI URL；在“系统设置”填写
有效地址后会自动应用，权威回读完成后应用才读取模型、GPU 与节点能力并开放上传和提交。

数据库默认位于 Director 项目根目录下的 `.data/database/director.sqlite3`，不随启动工作目录
变化。升级时若新位置尚不存在但仓库旧路径 `data/director.sqlite3` 存在，Director 会继续打开旧库，
避免一次升级直接出现空工作区。随后可在“系统设置 → 数据存储”中把当前库一致性迁移到默认位置，
或选择另一个已经存在且通过校验的 Director 数据库。迁移/切换成功后当前进程会停止普通写入；重启
Director 后新路径才生效。路径选择保存在数据库外的 `.data/database/storage.json`；可用
`DIRECTOR_STORAGE_CONFIG_PATH` 覆盖该配置文件位置，`DIRECTOR_DATABASE_PATH` 启动数据库覆盖
仍有更高优先级。页面会在操作前确认运行设置和时间线均已同步，并在成功后锁定编辑直到整页刷新；
每个写请求还携带本页启动时取得的数据库身份，因此只重启后端而保留旧页面也不能把旧库内容写进新库。

SQLite 的 `director.sqlite3-wal` 与 `director.sqlite3-shm` 是活动数据库的事务旁路文件，
`director.sqlite3.instance.lock` 是单实例锁元数据；它们都不是额外工作区，也不应在运行时单独
复制或删除。手工备份文件只有在明确恢复时才会使用，Director 不会自动把名字相似的备份当作
当前数据库。

本版没有登录鉴权，不要直接暴露到公网。跨机器访问前应在反向代理层增加 TLS、身份认证与来源限制。

## 开发与完整验证

```bash
UV_CACHE_DIR=/tmp/director-web-uv-cache uv sync --all-groups
UV_CACHE_DIR=/tmp/director-web-uv-cache uv run pytest backend/tests -q
cd frontend
npm ci
npm test
npm run build
```

轻量安装验证不需要模型或素材。完整原生 prompt 验证需要准备一个图片、一个 24fps 视频、旧 Turbo
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
