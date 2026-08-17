#!/usr/bin/env bash
# shellcheck shell=bash
# Automatic installation helpers for system packages, uv, Node.js and ffmpeg.

NODE_MIN_MAJOR=20
NODE_MIN_MINOR=19
NODE_ALT_MIN_MAJOR=22
NODE_ALT_MIN_MINOR=13

command_exists() { command -v "$1" >/dev/null 2>&1; }

pkg_install() {
  local packages=("$@")
  [[ "${#packages[@]}" -gt 0 ]] || return 0
  [[ "$PKG_MANAGER" == none ]] && {
    error "没有可用的包管理器，无法安装：${packages[*]}"
    return 1
  }
  [[ "$NO_SYSTEM_PACKAGES" == true ]] && {
    warn "已选择 --no-sudo，跳过系统包安装：${packages[*]}"
    return 1
  }
  info "安装系统包：${packages[*]}"
  local update_cmd=("${PKG_UPDATE[@]}") install_cmd=("${PKG_INSTALL[@]}")
  if [[ "$(id -u)" == 0 ]]; then
    [[ "${update_cmd[0]:-}" == sudo ]] && update_cmd=("${update_cmd[@]:1}")
    [[ "${install_cmd[0]:-}" == sudo ]] && install_cmd=("${install_cmd[@]:1}")
  fi
  "${update_cmd[@]}"
  "${install_cmd[@]}" "${packages[@]}"
}

ensure_system_packages() {
  local missing=() command_path
  for command in git curl tar; do
    command_path="$(command -v "$command" 2>/dev/null || true)"
    if [[ -z "$command_path" ]] || is_windows_interop_path "$command_path"; then
      missing+=("$command")
    else
      command_exists "$command" || missing+=("$command")
    fi
  done
  if [[ "$PKG_MANAGER" == apt ]]; then
    command_exists ffmpeg || missing+=("ffmpeg")
    [[ -f /etc/ssl/certs/ca-certificates.crt ]] || command_exists update-ca-certificates || missing+=("ca-certificates")
  elif [[ "$PKG_MANAGER" == dnf ]]; then
    command_exists ffmpeg || missing+=("ffmpeg-free")
    missing+=("ca-certificates")
  elif [[ "$PKG_MANAGER" == pacman ]]; then
    command_exists ffmpeg || missing+=("ffmpeg")
  fi
  [[ "${#missing[@]}" -gt 0 ]] || { ok "基础命令齐全：git curl tar"; return 0; }
  if pkg_install "${missing[@]}"; then
    ok "基础系统依赖已安装"
    return 0
  fi
  error "基础系统依赖安装失败；可重试或手动安装：${missing[*]}"
  return 1
}

node_version_compatible() {
  local version="$1" major minor
  IFS=. read -r major minor _ <<<"$version"
  ((major == NODE_MIN_MAJOR && minor >= NODE_MIN_MINOR)) \
    || ((major == NODE_ALT_MIN_MAJOR && minor >= NODE_ALT_MIN_MINOR)) \
    || ((major >= 24))
}

find_existing_node_bin_dir() {
  if [[ -n "${DIRECTOR_NODE_BIN_DIR:-}" ]]; then
    NODE_BIN_DIR="${DIRECTOR_NODE_BIN_DIR%/}"
    if is_windows_interop_path "$NODE_BIN_DIR/node"; then
      error "WSL 下不能使用 Windows 侧 Node.js：$NODE_BIN_DIR"
      return 1
    fi
    if [[ -x "$NODE_BIN_DIR/node" && -x "$NODE_BIN_DIR/npm" ]]; then
      return 0
    fi
    error "DIRECTOR_NODE_BIN_DIR 无效：$NODE_BIN_DIR"
    return 1
  fi
  if command_exists node && command_exists npm; then
    local node_path npm_path node_dir npm_dir
    node_path="$(command -v node)"
    npm_path="$(command -v npm)"
    if is_windows_interop_path "$node_path" || is_windows_interop_path "$npm_path"; then
      warn "PATH 中的 node/npm 来自 Windows；WSL2 下将安装 Linux 版 Node.js"
      NODE_BIN_DIR=""
      return 1
    fi
    node_dir="$(cd -- "$(dirname -- "$node_path")" && pwd)"
    npm_dir="$(cd -- "$(dirname -- "$npm_path")" && pwd)"
    if [[ "$node_dir" == "$npm_dir" ]]; then
      NODE_BIN_DIR="$node_dir"
      return 0
    fi
  fi
  NODE_BIN_DIR=""
  return 1
}

install_node_portable() {
  local arch
  case "$(uname -m)" in
    x86_64) arch=x64 ;;
    aarch64|arm64) arch=arm64 ;;
    *) error "不支持的 CPU 架构：$(uname -m)"; return 1 ;;
  esac
  local dist_url="https://nodejs.org/dist/latest-v22.x"
  local work_dir="$STATE_DIR/node-download"
  mkdir -p "$work_dir"
  info "从 $dist_url 获取 Node.js 22 LTS"
  local checksums archive_name checksum
  checksums="$(curl -fsSL --retry 3 "$dist_url/SHASUMS256.txt")" || {
    error "下载 Node.js SHASUMS256.txt 失败"
    return 1
  }
  archive_name="$(awk -v pattern="linux-${arch}.tar.xz" '$2 ~ pattern {print $2; exit}' <<<"$checksums")"
  [[ -n "$archive_name" ]] || { error "未找到 linux-${arch} 的 Node.js 包"; return 1; }
  checksum="$(awk -v name="$archive_name" '$2 == name {print $1}' <<<"$checksums")"
  [[ -n "$checksum" ]] || { error "未找到 $archive_name 校验值"; return 1; }

  [[ -f "$work_dir/$archive_name" ]] || curl -fL --retry 3 -o "$work_dir/$archive_name" "$dist_url/$archive_name" || {
    error "下载 Node.js 失败"
    return 1
  }
  printf '%s  %s\n' "$checksum" "$archive_name" >"$work_dir/SHASUMS256.txt"
  (cd "$work_dir" && sha256sum -c SHASUMS256.txt) || { error "Node.js 校验失败"; return 1; }

  rm -rf "$TOOLS_DIR/node"
  mkdir -p "$TOOLS_DIR"
  tar -xJf "$work_dir/$archive_name" -C "$TOOLS_DIR" || { error "Node.js 解压失败"; return 1; }
  local extracted
  extracted="$(find "$TOOLS_DIR" -maxdepth 1 -type d -name 'node-v*' | head -n 1)"
  [[ -n "$extracted" ]] || { error "Node.js 解压目录不存在"; return 1; }
  mv "$extracted" "$TOOLS_DIR/node"
  mkdir -p "$TOOLS_BIN_DIR"
  ln -sfn "$TOOLS_DIR/node/bin/node" "$TOOLS_BIN_DIR/node"
  ln -sfn "$TOOLS_DIR/node/bin/npm" "$TOOLS_BIN_DIR/npm"
  ln -sfn "$TOOLS_DIR/node/bin/npx" "$TOOLS_BIN_DIR/npx"
  rm -f "$work_dir/SHASUMS256.txt"
  NODE_BIN_DIR="$TOOLS_BIN_DIR"
  ok "Node.js 已安装到 $TOOLS_BIN_DIR"
}

ensure_node() {
  if [[ -n "${DIRECTOR_NODE_BIN_DIR:-}" ]]; then
    NODE_BIN_DIR="${DIRECTOR_NODE_BIN_DIR%/}"
    [[ -x "$NODE_BIN_DIR/node" && -x "$NODE_BIN_DIR/npm" ]] || {
      error "指定的 DIRECTOR_NODE_BIN_DIR 中缺少 node/npm：$NODE_BIN_DIR"
      return 1
    }
    find_existing_node_bin_dir || return 1
  fi
  if find_existing_node_bin_dir; then
    local version
    version="$("$NODE_BIN_DIR/node" -p 'process.versions.node')"
    if node_version_compatible "$version"; then
      ok "Node.js $version（$NODE_BIN_DIR）"
      return 0
    fi
    warn "现有 Node.js $version 不满足 ^20.19、^22.13 或 >=24"
    if ! ask_yes_no "是否安装 Node.js 22 LTS 到项目 .tools？" y; then
      error "Node.js 版本不兼容"
      return 1
    fi
  else
    if ! ask_yes_no "未找到兼容的 Node.js，是否自动安装 Node.js 22 LTS 到项目 .tools？" y; then
      error "Node.js 不可用"
      return 1
    fi
  fi
  install_node_portable
}

check_ffmpeg_features() {
  command_exists ffmpeg && command_exists ffprobe || return 1
  local encoders filters
  encoders="$(ffmpeg -hide_banner -encoders 2>&1)" || return 1
  filters="$(ffmpeg -hide_banner -filters 2>&1)" || return 1
  for encoder in libx264 aac; do
    grep -Eq "[[:space:]]${encoder}[[:space:]]" <<<"$encoders" || return 1
  done
  for filter in fps scale select aresample; do
    grep -Eq "[[:space:]]${filter}[[:space:]]" <<<"$filters" || return 1
  done
  return 0
}

install_ffmpeg_static() {
  [[ "$(uname -m)" == x86_64 ]] || {
    error "没有 sudo 且不是 x86_64，无法自动下载静态 ffmpeg；请手动安装 ffmpeg"
    return 1
  }
  local url="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
  local archive="$TOOLS_DIR/ffmpeg-release-amd64-static.tar.xz"
  local extracted_dir="$TOOLS_DIR/ffmpeg-static"
  info "下载静态 ffmpeg：$url"
  [[ -f "$archive" ]] || curl -fL --retry 3 -o "$archive" "$url" || {
    error "静态 ffmpeg 下载失败"
    return 1
  }
  rm -rf "$extracted_dir"
  mkdir -p "$extracted_dir"
  tar -xJf "$archive" -C "$extracted_dir" || { error "静态 ffmpeg 解压失败"; return 1; }
  local bin_dir
  bin_dir="$(find "$extracted_dir" -maxdepth 2 -type f -name ffmpeg | head -n 1 | xargs -r dirname)"
  [[ -n "$bin_dir" ]] || { error "静态 ffmpeg 包中未找到 ffmpeg"; return 1; }
  mkdir -p "$TOOLS_BIN_DIR"
  ln -sfn "$bin_dir/ffmpeg" "$TOOLS_BIN_DIR/ffmpeg"
  ln -sfn "$bin_dir/ffprobe" "$TOOLS_BIN_DIR/ffprobe"
  export PATH="$TOOLS_BIN_DIR:$PATH"
  ok "静态 ffmpeg 已安装到 $TOOLS_BIN_DIR"
}

ensure_ffmpeg() {
  local ffmpeg_path
  ffmpeg_path="$(command -v ffmpeg 2>/dev/null || true)"
  if check_ffmpeg_features && [[ -z "$ffmpeg_path" ]] || ! is_windows_interop_path "$ffmpeg_path"; then
    ok "ffmpeg/ffprobe 可用且包含 libx264/aac 与所需 filters"
    return 0
  fi
  if is_windows_interop_path "$ffmpeg_path"; then
    warn "PATH 中的 ffmpeg 来自 Windows；WSL2 下将安装 Linux 版 ffmpeg"
  fi
  if [[ "$PKG_MANAGER" != none && "$NO_SYSTEM_PACKAGES" != true ]]; then
    if pkg_install ffmpeg; then
      export PATH="$(command -v ffmpeg | xargs dirname):$PATH"
      check_ffmpeg_features && { ok "系统 ffmpeg 满足要求"; return 0; }
      warn "系统 ffmpeg 缺少所需 encoder/filter"
    fi
  fi
  if ! ask_yes_no "是否下载静态 ffmpeg 到项目 .tools？" y; then
    error "ffmpeg/ffprobe 不满足 Director 要求"
    return 1
  fi
  install_ffmpeg_static && check_ffmpeg_features
}

ensure_uv() {
  if command_exists uv && ! is_windows_interop_path "$(command -v uv)"; then
    UV_BIN="$(command -v uv)"
    ok "uv 已可用：$("$UV_BIN" --version 2>/dev/null || true)"
    return 0
  fi
  if command_exists uv; then
    warn "PATH 中的 uv 来自 Windows；WSL2 下将安装 Linux 版 uv"
  fi
  if [[ -x "$TOOLS_BIN_DIR/uv" ]]; then
    UV_BIN="$TOOLS_BIN_DIR/uv"
    ok "uv 已可用：$("$UV_BIN" --version 2>/dev/null || true)"
    return 0
  fi
  if ! ask_yes_no "未找到 uv，是否自动安装到项目 .tools？" y; then
    error "uv 不可用"
    return 1
  fi
  mkdir -p "$TOOLS_BIN_DIR" "$STATE_DIR"
  local installer="$STATE_DIR/uv-install.sh"
  info "安装 uv：https://astral.sh/uv/install.sh"
  curl -fLsS --retry 3 -o "$installer" https://astral.sh/uv/install.sh || {
    error "uv 安装脚本下载失败"
    return 1
  }
  env UV_INSTALL_DIR="$TOOLS_BIN_DIR" UV_NO_MODIFY_PATH=1 XDG_CONFIG_HOME="$TOOLS_DIR/config" sh "$installer" || {
    error "uv 安装失败"
    return 1
  }
  rm -f "$installer"
  [[ -x "$TOOLS_BIN_DIR/uv" ]] || { error "uv 安装后不可执行"; return 1; }
  UV_BIN="$TOOLS_BIN_DIR/uv"
  ok "uv 已安装：$("$UV_BIN" --version)"
}
