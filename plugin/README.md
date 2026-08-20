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

## 媒体工具（ffmpeg）

素材探测、转码和拼接需要 ffmpeg 与 ffprobe（含 libx264/aac）。缺失时可在“系统设置 → 媒体工具”
一键安装（static-ffmpeg，立即生效无需重启），或按平台手动安装。

## 数据位置

数据库固定在 ComfyUI `user/directordeck/database/directordeck.sqlite3`；素材与生成输出由
ComfyUI 的 input/output 目录管理。插件目录内不保存任何用户数据，可直接升级覆盖。

## 许可

GPL-3.0-only。第三方组件见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
