# DirectorDeck

把 [Director](https://github.com/JYE-HC/Director-WebUI) 长视频导演台嵌入 ComfyUI：
面向 MiniMax H3 的统一时间线创作——资产库、逐段 FL2VA/Ref2VA 配方推导、任务编排、
逐段进度与实时预览、长片拼接，全部在 ComfyUI 进程内完成。

## 安装

**ComfyUI Manager（推荐）**：搜索 `DirectorDeck` 一键安装，重启 ComfyUI。

**手动安装**：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/JYE-HC/DirectorDeck.git
# 用运行 ComfyUI 的 Python 环境安装依赖：
/path/to/ComfyUI/.venv/bin/python -m pip install -r DirectorDeck/requirements.txt
```

重启 ComfyUI 后，侧栏出现 **Director** 面板，点击“打开 Director”即可在新标签页进入导演台
（页面托管于同源的 `/directordeck/`）。ComfyUI 地址自动指向本实例，无需配置。

建议使用 ComfyUI 0.33.0 或更新版本。旧版本或无法解析的版本只会产生兼容性警告，
不会阻止 DirectorDeck 启动；实际不兼容行为会在导入、提示词校验或执行时按原始错误报告。

## 安全边界

DirectorDeck 面向本机单一可信用户或可信私网，没有内置登录、授权边界或 TLS 终止。
不要把 ComfyUI 或 DirectorDeck 直接暴露到公网。跨机器访问前，必须在反向代理层增加 TLS、
身份认证与来源限制。漏洞报告和诊断信息处理要求见 [SECURITY.md](SECURITY.md)。

## 多卡推理（RayLight，仅 Linux）

默认不安装多卡组件。在“系统设置 → 多卡推理”开启开关，按提示一键安装
（ray/xfuser，torch 版本保持不变），重启 ComfyUI 后，GPU 池配置 2 张及以上逻辑卡即自动
使用 RayLight。也可手动安装后重启：

```bash
/path/to/ComfyUI/.venv/bin/python -m pip install -r DirectorDeck/requirements-raylight.txt
```

Director 多卡工作流只使用插件内置并维护的 `DirectorDeck-RayLight`，并通过
`DirectorDeckRay*` 专属节点名注册；其本地代码和 Ray worker 统一使用私有
`directordeck_raylight` Python 命名空间。外部 `custom_nodes/raylight` 即使存在也不会被
Director 导入、跳过内置实现或选入 Director 的工作流。

## Standard LoRA 加载节点

DirectorDeck 不打包或维护第三方 Standard LoRA 加载节点。请先将所选加载节点安装到
ComfyUI，再在 Director 的系统设置中为底模与 LoRA 建立精确映射。映射是用户权威：
DirectorDeck 不按模块名、接口切片或实现指纹拒绝用户安装或修改过的加载节点；真实导入、
提示词校验与执行错误由 ComfyUI/Director 原样报告。

## 媒体工具（ffmpeg）

素材探测、转码和拼接需要 ffmpeg 与 ffprobe（含 libx264/aac）。缺失时可在“系统设置 → 媒体工具”
一键安装（static-ffmpeg，立即生效无需重启），或按平台手动安装。

## 数据位置

数据库固定在 ComfyUI `user/directordeck/database/directordeck.sqlite3`；素材与生成输出由
ComfyUI 的 input/output 目录管理。插件目录内不保存任何用户数据，可直接升级覆盖。

## 许可

GPL-3.0-only。第三方组件见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
