#!/usr/bin/env bash
# shellcheck shell=bash
# Platform detection helpers for Linux native and WSL2.

IS_WSL=false
WSL_VERSION=0
PKG_MANAGER=""
PKG_INSTALL=()
HAS_SUDO=false
HAS_SYSTEMD=false
HAS_USER_SYSTEMD=false
PROJECT_ON_WINDOWS_FS=false
NVIDIA_SMI=false
WSL_INTEROP=false

detect_wsl() {
  IS_WSL=false
  WSL_VERSION=0
  if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
    IS_WSL=true
    if uname -r 2>/dev/null | grep -qi 'WSL2'; then
      WSL_VERSION=2
    else
      WSL_VERSION=1
    fi
  fi
  [[ -n "${WSL_DISTRO_NAME:-}" ]] && IS_WSL=true
  if [[ "$IS_WSL" == true && "$WSL_VERSION" == 0 ]]; then
    WSL_VERSION=2
  fi
  if [[ "$IS_WSL" == true ]]; then
    ok "检测到 WSL${WSL_VERSION}：${WSL_DISTRO_NAME:-unknown distro}"
  else
    ok "检测到原生 Linux"
  fi
}

detect_package_manager() {
  if command -v apt-get >/dev/null 2>&1; then
    PKG_MANAGER=apt
    PKG_UPDATE=(sudo apt-get update -y)
    PKG_INSTALL=(sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends)
  elif command -v dnf >/dev/null 2>&1; then
    PKG_MANAGER=dnf
    PKG_UPDATE=(sudo dnf -y makecache)
    PKG_INSTALL=(sudo dnf -y install)
  elif command -v pacman >/dev/null 2>&1; then
    PKG_MANAGER=pacman
    PKG_UPDATE=(sudo pacman -Sy --noconfirm)
    PKG_INSTALL=(sudo pacman -S --needed --noconfirm)
  elif command -v zypper >/dev/null 2>&1; then
    PKG_MANAGER=zypper
    PKG_UPDATE=(sudo zypper --non-interactive refresh)
    PKG_INSTALL=(sudo zypper --non-interactive install)
  else
    PKG_MANAGER=none
    PKG_UPDATE=()
    PKG_INSTALL=()
  fi
  [[ "$PKG_MANAGER" == none ]] && return 1
  ok "包管理器：$PKG_MANAGER"
  return 0
}

detect_sudo() {
  if [[ "$(id -u)" == 0 ]]; then
    HAS_SUDO=true
    ok "以 root 运行，可安装系统包"
    return 0
  fi
  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    HAS_SUDO=true
    ok "sudo 可用（免密或已有凭据）"
    return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    HAS_SUDO=true
    warn "sudo 需要交互输入密码"
    return 0
  fi
  HAS_SUDO=false
  warn "sudo 不可用；系统包安装将被跳过，uv/node/ffmpeg 将尽量装到项目 .tools"
  return 0
}

detect_systemd() {
  HAS_SYSTEMD=false
  HAS_USER_SYSTEMD=false
  if command -v systemctl >/dev/null 2>&1 && command -v systemd-run >/dev/null 2>&1; then
    if [[ -d /run/systemd/system ]]; then
      HAS_SYSTEMD=true
      if timeout 3 systemctl --user show-environment >/dev/null 2>&1; then
        HAS_USER_SYSTEMD=true
        ok "systemd user bus 可用，可使用现有 director.sh"
      else
        warn "systemd 存在但 user bus 不可用，将使用内置 supervisor"
      fi
    else
      warn "systemd 工具存在但 PID1 不是 systemd，将使用内置 supervisor"
    fi
  else
    warn "未检测到 systemd 用户服务工具，将使用内置 supervisor"
  fi
  return 0
}

detect_project_fs() {
  PROJECT_ON_WINDOWS_FS=false
  if [[ "$IS_WSL" == true && "$SCRIPT_DIR" == /mnt/* ]]; then
    PROJECT_ON_WINDOWS_FS=true
    warn "项目位于 Windows 文件系统：$SCRIPT_DIR"
    warn "建议迁移到 WSL ext4（例如 ~/director-web）以保障 SQLite/WAL、flock 与 npm/Vite 性能"
  fi
  return 0
}

detect_gpu() {
  NVIDIA_SMI=false
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    NVIDIA_SMI=true
    ok "检测到 NVIDIA GPU"
  elif [[ "$IS_WSL" == true ]]; then
    warn "WSL 内未检测到 nvidia-smi；若使用本地 ComfyUI GPU 生成，请先确认 Windows 侧 NVIDIA 驱动与 WSL CUDA 支持"
  else
    warn "未检测到 nvidia-smi；本地 ComfyUI 可能只能做 CPU 验证"
  fi
  return 0
}

detect_interop() {
  WSL_INTEROP=false
  [[ -f /proc/sys/fs/binfmt_misc/WSLInterop ]] && WSL_INTEROP=true
  return 0
}

windows_path_to_wsl() {
  local value="$1"
  value="$(printf '%s' "$value" | tr -d '\r')"
  if command -v wslpath >/dev/null 2>&1 && [[ -n "$value" ]]; then
    wslpath -u "$value" 2>/dev/null || true
    return
  fi
  # Fallback for WSL without wslpath: C:\ -> /mnt/c, C:/ -> /mnt/c
  if [[ "$value" =~ ^([A-Za-z]):\\?(.*)$ ]]; then
    local drive="${BASH_REMATCH[1],,}" rest="${BASH_REMATCH[2]}"
    rest="${rest//\\//}"
    printf '/mnt/%s/%s\n' "$drive" "${rest#/}"
    return
  fi
  printf '%s\n' "$value"
}

project_dirname() {
  local path="$SCRIPT_DIR"
  path="${path%/}"
  printf '%s\n' "${path##*/}"
}

default_comfyui_port() {
  printf '%s\n' "${COMFYUI_PORT:-28188}"
}

host_ip() {
  hostname -I 2>/dev/null | awk '{print $1}'
}


is_windows_interop_path() {
  local value="$1"
  [[ "$IS_WSL" == true ]] || return 1
  case "$value" in
    /mnt/*) return 0 ;;
    *.exe|*.EXE) return 0 ;;
  esac
  return 1
}
