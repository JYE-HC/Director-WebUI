#!/usr/bin/env bash
# shellcheck shell=bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$SCRIPT_DIR/.director-install"
STATE_FILE="$STATE_DIR/state.env"
ENV_FILE="$STATE_DIR/env.sh"
SETUP_LOG="$STATE_DIR/setup.log"
LOCK_DIR="$STATE_DIR/setup.lock"
TOOLS_DIR="$SCRIPT_DIR/.tools"
TOOLS_BIN_DIR="$TOOLS_DIR/bin"

source "$SCRIPT_DIR/scripts/bootstrap/common.sh"
source "$SCRIPT_DIR/scripts/bootstrap/detect.sh"
source "$SCRIPT_DIR/scripts/bootstrap/install_tools.sh"
source "$SCRIPT_DIR/scripts/bootstrap/comfyui.sh"
source "$SCRIPT_DIR/scripts/bootstrap/director.sh"
source "$SCRIPT_DIR/scripts/bootstrap/launcher.sh"

COMMAND="install"
ORIGINAL_ARGS=()
ASSUME_YES=false
DRY_RUN=false
NO_SYSTEM_PACKAGES=false
SKIP_COMFYUI=false
SKIP_COMFYUI_DEPS=false
START_AFTER_INSTALL=false
START_COMFYUI_AFTER_INSTALL=false
REQUIRE_COMFY_ONLINE=false
DIRECTOR_SKIP_RELOCATION=false
COMFYUI_ROOT_OPTION=""
WINDOWS_COMFYUI_ROOT_OPTION=""
COMFYUI_PYTHON_OPTION=""
DIRECTOR_NODE_BIN_DIR="${DIRECTOR_NODE_BIN_DIR:-}"
DIRECTOR_BIND_HOST=""
DIRECTOR_BACKEND_PORT=8787
DIRECTOR_FRONTEND_PORT=4173
UV_BIN=""

usage() {
  cat <<'EOF'
用法：
  ./bootstrap.sh [install] [选项]     # 自动安装 Linux / WSL2 环境下的 Director Web
  ./bootstrap.sh check [选项]         # 只检查环境，不安装
  ./bootstrap.sh verify [选项]        # 只验证已安装内容
  ./bootstrap.sh start|stop|restart|status|logs   # 同时管理 Director 与本地 ComfyUI
  ./bootstrap.sh start-director|stop-director|restart-director|status-director|logs-director [director.sh 选项]
  ./bootstrap.sh start-comfyui|stop-comfyui|restart-comfyui|status-comfyui|logs-comfyui
  ./bootstrap.sh reset                # 清空安装状态，不删除 ComfyUI/数据

常用选项：
  -y, --yes                     使用推荐默认值，不再逐项询问
      --dry-run                 只显示检测结果与安装计划
      --resume                  从上次失败步骤继续
      --from STEP               从指定步骤开始
      --only STEP               只执行指定步骤（已完成也强制重跑）
      --skip STEP[,STEP...]     跳过指定步骤
      --no-sudo                 禁止安装系统包；uv/node/ffmpeg 尽量装入项目 .tools
      --node-bin-dir PATH       使用已有 node 与 npm 目录
      --listen-host HOST        设置 Director 前后端监听地址
      --backend-port PORT       后端端口（默认 8787）
      --frontend-port PORT      前端端口（默认 4173）
      --comfyui-root PATH       使用已有 ComfyUI 根目录
      --comfy-python PATH       ComfyUI 使用的 Python（默认 <root>/.venv/bin/python）
      --windows-comfyui-root PATH WSL2 下复用的 Windows ComfyUI 根目录（用于共享 models）
      --comfyui-ref REF         自动 clone 的版本：latest|tested|<tag/sha>
      --skip-comfyui            不安装本地 ComfyUI，连接远程
      --comfyui-url URL         远程 ComfyUI 地址
      --comfyui-port PORT       WSL 内本地 ComfyUI 端口（默认 28188）
      --skip-comfyui-deps       只创建 ComfyUI venv，不安装 requirements
      --start                   安装成功后启动 Director
      --start-comfyui           安装成功后启动本地 ComfyUI
      --require-online           ComfyUI 在线检查失败则安装失败
      --skip-relocation         WSL2 下即使项目在 /mnt/c 也不自动迁移
  -h, --help                    显示帮助
EOF
}

parse_args() {
  case "${1:-}" in
    start|stop|restart|status|logs|start-director|stop-director|restart-director|status-director|logs-director|start-comfyui|stop-comfyui|restart-comfyui|status-comfyui|logs-comfyui)
      COMMAND="$1"
      ORIGINAL_ARGS=("${@:2}")
      return 0
      ;;
  esac

  if (($# == 0)) || [[ "$1" == -* ]]; then
    set -- install "$@"
  fi
  COMMAND="$1"
  shift
  ORIGINAL_ARGS=("$@")

  while (($#)); do
    case "$1" in
      -y|--yes) ASSUME_YES=true; shift ;;
      --dry-run) DRY_RUN=true; shift ;;
      --resume) shift ;;
      --from) (($# >= 2)) || die "缺少 --from 参数"; FROM_STEP="$2"; shift 2 ;;
      --only) (($# >= 2)) || die "缺少 --only 参数"; ONLY_STEP="$2"; shift 2 ;;
      --skip) (($# >= 2)) || die "缺少 --skip 参数"; add_skip_steps "$2"; shift 2 ;;
      --no-sudo) NO_SYSTEM_PACKAGES=true; shift ;;
      --node-bin-dir) (($# >= 2)) || die "缺少 --node-bin-dir 参数"; DIRECTOR_NODE_BIN_DIR="$2"; shift 2 ;;
      --listen-host|--host) (($# >= 2)) || die "缺少 $1 参数"; DIRECTOR_BIND_HOST="$2"; shift 2 ;;
      --backend-port) (($# >= 2)) || die "缺少 --backend-port 参数"; DIRECTOR_BACKEND_PORT="$2"; shift 2 ;;
      --frontend-port) (($# >= 2)) || die "缺少 --frontend-port 参数"; DIRECTOR_FRONTEND_PORT="$2"; shift 2 ;;
      --comfyui-root|--comfyui) (($# >= 2)) || die "缺少 $1 参数"; COMFYUI_ROOT_OPTION="$2"; shift 2 ;;
      --comfy-python) (($# >= 2)) || die "缺少 --comfy-python 参数"; COMFYUI_PYTHON_OPTION="$2"; shift 2 ;;
      --windows-comfyui-root) (($# >= 2)) || die "缺少 --windows-comfyui-root 参数"; WINDOWS_COMFYUI_ROOT_OPTION="$2"; WINDOWS_COMFYUI_ROOT="$2"; shift 2 ;;
      --comfyui-ref) (($# >= 2)) || die "缺少 --comfyui-ref 参数"; COMFYUI_REF="$2"; shift 2 ;;
      --skip-comfyui) SKIP_COMFYUI=true; shift ;;
      --comfyui-url) (($# >= 2)) || die "缺少 --comfyui-url 参数"; COMFYUI_URL="$2"; shift 2 ;;
      --comfyui-port) (($# >= 2)) || die "缺少 --comfyui-port 参数"; COMFYUI_PORT="$2"; shift 2 ;;
      --skip-comfyui-deps) SKIP_COMFYUI_DEPS=true; shift ;;
      --start) START_AFTER_INSTALL=true; shift ;;
      --start-comfyui) START_COMFYUI_AFTER_INSTALL=true; shift ;;
      --require-online) REQUIRE_COMFY_ONLINE=true; shift ;;
      --skip-relocation) DIRECTOR_SKIP_RELOCATION=true; shift ;;
      -h|--help) usage; exit 0 ;;
      *) die "未知选项：$1" 64 ;;
    esac
  done

  if [[ "$COMMAND" == install ]]; then
    [[ -z "${FROM_STEP:-}" || -z "${ONLY_STEP:-}" ]] || die "--from 与 --only 不能同时使用" 64
  fi
}

step_detect_platform() {
  [[ "$(uname -s)" == Linux ]] || { error "当前只支持 Linux / WSL2"; return 1; }
  detect_wsl
  detect_interop
  detect_package_manager || true
  detect_sudo
  detect_systemd
  detect_project_fs
  detect_gpu
  state_set IS_WSL "$IS_WSL"
  state_set WSL_VERSION "$WSL_VERSION"
  state_set PKG_MANAGER "$PKG_MANAGER"
  state_set HAS_USER_SYSTEMD "$HAS_USER_SYSTEMD"
  state_set NVIDIA_SMI "$NVIDIA_SMI"
  state_flush
  return 0
}

step_install_system_packages() {
  if [[ "$NO_SYSTEM_PACKAGES" == true ]]; then
    warn "已选择 --no-sudo，跳过系统包安装"
    return 0
  fi
  ensure_system_packages
}

step_install_uv() {
  ensure_uv
  state_set UV_BIN "$UV_BIN"
  state_flush
}

step_install_node() {
  ensure_node
  state_set NODE_BIN_DIR "$NODE_BIN_DIR"
  state_flush
}

step_install_ffmpeg() {
  ensure_ffmpeg
}

declare -A STEP_TITLES=(
  [detect_platform]="检测 Linux / WSL2 运行平台"
  [resolve_layout]="检查项目文件系统位置"
  [select_comfyui_mode]="选择 ComfyUI 安装方式"
  [install_system_packages]="安装基础系统依赖"
  [install_uv]="安装 uv"
  [install_node]="安装 Node.js"
  [install_ffmpeg]="安装/检查 ffmpeg"
  [prepare_comfyui]="准备 ComfyUI"
  [install_comfyui_deps]="安装 ComfyUI 依赖"
  [install_director]="安装 Director 与 custom nodes"
  [verify_offline]="离线验证安装"
  [verify_online_optional]="在线验证 ComfyUI"
  [configure_launcher]="生成启动配置"
  [start_services_optional]="启动服务"
)

STEP_QUEUE=(
  detect_platform
  resolve_layout
  select_comfyui_mode
  install_system_packages
  install_uv
  install_node
  install_ffmpeg
  prepare_comfyui
  install_comfyui_deps
  install_director
  verify_offline
  verify_online_optional
  configure_launcher
  start_services_optional
)

print_plan() {
  section "安装计划"
  local step
  for step in "${STEP_QUEUE[@]}"; do
    printf '  - %-28s %s\n' "$step" "${STEP_TITLES[$step]:-$step}"
  done
  info ""
  if [[ -n "$COMFYUI_ROOT_OPTION" ]]; then
    info "ComfyUI 模式：existing（$COMFYUI_ROOT_OPTION）"
  elif [[ "$SKIP_COMFYUI" == true ]]; then
    info "ComfyUI 模式：skip（远程 $COMFYUI_URL）"
  else
    info "ComfyUI 模式：auto（$COMFYUI_REF）"
  fi
}

run_install() {
  mkdir -p "$STATE_DIR"
  load_state
  acquire_lock || exit 1
  if [[ "$DRY_RUN" == true ]]; then
    STATE_READONLY=true
    step_detect_platform
    print_plan
    ok "dry-run：未写入任何文件"
    return 0
  fi

  exec > >(tee -a "$SETUP_LOG") 2>&1
  # Platform detection is read-only, but its results (PKG_UPDATE/PKG_INSTALL
  # arrays, PROJECT_ON_WINDOWS_FS, …) live only in this process and cannot be
  # restored from state.env. Re-run it on every install invocation or resumed
  # steps would see empty package-manager commands.
  step_detect_platform || exit 1
  mark_step detect_platform done
  run_chain || exit $?
  section "安装完成"
  ok "Director Web 安装流程已完成"
  if [[ "$COMFYUI_MODE" != skip && -n "$COMFYUI_ROOT" ]]; then
    info "ComfyUI：$COMFYUI_ROOT"
    info "启动全部服务：./bootstrap.sh start（Director 与本地 ComfyUI）"
  else
    info "启动 Director：./bootstrap.sh start"
  fi
  if [[ "$IS_WSL" == true ]]; then
    info "浏览器输入 http://localhost:${DIRECTOR_FRONTEND_PORT:-4173} 访问 Director-WebUI（Windows 侧）"
    info "WSL 内则输入 http://127.0.0.1:${DIRECTOR_FRONTEND_PORT:-4173}"
  else
    info "浏览器输入 http://127.0.0.1:${DIRECTOR_FRONTEND_PORT:-4173} 访问 Director-WebUI"
  fi
}

run_check() {
  mkdir -p "$STATE_DIR"
  STATE_READONLY=true
  step_detect_platform
  section "环境检查（只读）"
  local failed=0
  if command_exists git; then ok "git：$(git --version)"; else error "缺少 git"; failed=1; fi
  if command_exists curl; then ok "curl：$(curl --version | head -n 1)"; else error "缺少 curl"; failed=1; fi
  ensure_uv_check || failed=1
  ensure_node_check || failed=1
  ensure_ffmpeg_check || failed=1
  print_plan
  if ((failed)); then
    error "环境检查存在失败项"
    return 1
  fi
  ok "环境检查完成"
}

ensure_uv_check() {
  if command_exists uv; then ok "uv：$(uv --version)"; return 0; fi
  warn "缺少 uv，安装时会自动补装"
  return 1
}
ensure_node_check() {
  if find_existing_node_bin_dir; then
    local version="$("$NODE_BIN_DIR/node" -p 'process.versions.node')"
    if node_version_compatible "$version"; then ok "Node.js：$version"; return 0; fi
    warn "Node.js $version 版本不兼容"
  else
    warn "缺少兼容 Node.js，安装时会自动补装"
  fi
  return 1
}
ensure_ffmpeg_check() {
  if check_ffmpeg_features; then ok "ffmpeg/ffprobe 满足要求"; return 0; fi
  warn "ffmpeg 缺失或不完整，安装时会自动补装"
  return 1
}

run_verify() {
  mkdir -p "$STATE_DIR"
  load_state
  load_env_file
  if [[ -n "$COMFYUI_ROOT_OPTION" ]]; then
    COMFYUI_MODE=existing
    COMFYUI_ROOT="$COMFYUI_ROOT_OPTION"
  elif [[ "$SKIP_COMFYUI" == true ]]; then
    COMFYUI_MODE=skip
    [[ -n "$COMFYUI_URL" ]] && DIRECTOR_COMFYUI_URL="$COMFYUI_URL"
  elif [[ -z "${COMFYUI_ROOT:-}" && -n "${DIRECTOR_COMFYUI_ROOT:-}" ]]; then
    COMFYUI_MODE=existing
    COMFYUI_ROOT="$DIRECTOR_COMFYUI_ROOT"
  fi
  NODE_BIN_DIR="${DIRECTOR_NODE_BIN_DIR:-${NODE_BIN_DIR:-}}"
  [[ -x "$SCRIPT_DIR/.venv/bin/director-web" ]] || { error "后端未安装"; return 1; }
  [[ -d "$SCRIPT_DIR/frontend/dist" ]] || { error "前端未构建"; return 1; }
  "$SCRIPT_DIR/.venv/bin/python" -c 'import director' || { error "director 包不可导入"; return 1; }
  if [[ "$COMFYUI_MODE" != skip && -n "$COMFYUI_ROOT" ]]; then
    step_verify_offline
  fi
  ok "验证完成"
}

has_local_comfyui() {
  [[ "${COMFYUI_MODE:-skip}" != skip ]]
}

run_combined_service_command() {
  local py="$SCRIPT_DIR/.venv/bin/python"
  local supervisor=("$py" "$SCRIPT_DIR/tools/director_supervisor.py")
  case "$COMMAND" in
    start)
      local rc=0 director_rc=0
      if has_local_comfyui; then
        [[ -x "$py" ]] || die "后端 Python 尚未安装" 4
        "${supervisor[@]}" start-comfyui || rc=1
      fi
      "$SCRIPT_DIR/director.sh" start "${ORIGINAL_ARGS[@]}" || { rc=1; director_rc=1; }
      if [[ $director_rc -eq 0 ]] && [[ -n "$(comfyui_seed_url)" ]]; then
        seed_comfyui_url_if_unset || warn "ComfyUI 地址预置未完成；请稍后在系统设置中手动填写"
      fi
      return "$rc"
      ;;
    stop)
      local rc=0
      "$SCRIPT_DIR/director.sh" stop || rc=1
      if has_local_comfyui; then
        [[ -x "$py" ]] || die "后端 Python 尚未安装" 4
        "${supervisor[@]}" stop-comfyui || rc=1
      fi
      return "$rc"
      ;;
    restart)
      local rc=0
      if has_local_comfyui; then
        [[ -x "$py" ]] || die "后端 Python 尚未安装" 4
        "${supervisor[@]}" restart-comfyui || rc=1
      fi
      "$SCRIPT_DIR/director.sh" restart "${ORIGINAL_ARGS[@]}" || rc=1
      return "$rc"
      ;;
    status)
      "$SCRIPT_DIR/director.sh" status || true
      if has_local_comfyui; then
        [[ -x "$py" ]] || die "后端 Python 尚未安装" 4
        "${supervisor[@]}" status-comfyui || true
      fi
      return 0
      ;;
    logs)
      local target="${ORIGINAL_ARGS[0]:-}"
      case "$target" in
        backend|frontend)
          exec "$SCRIPT_DIR/director.sh" logs "$target" ;;
        comfyui)
          [[ -x "$py" ]] || die "后端 Python 尚未安装" 4
          exec "${supervisor[@]}" logs-comfyui ;;
        "")
          local data_dir="$SCRIPT_DIR/data"
          local files=()
          [[ -f "$data_dir/director-backend.log" ]] && files+=("$data_dir/director-backend.log")
          [[ -f "$data_dir/director-frontend.log" ]] && files+=("$data_dir/director-frontend.log")
          if has_local_comfyui && [[ -f "$data_dir/comfyui.log" ]]; then
            files+=("$data_dir/comfyui.log")
          fi
          ((${#files[@]})) || die "暂无日志文件；服务尚未启动过" 4
          exec tail -F "${files[@]}"
          ;;
        *) die "logs 只接受 backend、frontend 或 comfyui" 64 ;;
      esac
      ;;
  esac
}

run_service_command() {
  load_env_file
  load_state
  case "$COMMAND" in
    *-comfyui)
      [[ -x "$SCRIPT_DIR/.venv/bin/python" ]] || die "后端 Python 尚未安装" 4
      exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/tools/director_supervisor.py" "$COMMAND" "${ORIGINAL_ARGS[@]}"
      ;;
    *-director)
      exec "$SCRIPT_DIR/director.sh" "${COMMAND%-director}" "${ORIGINAL_ARGS[@]}"
      ;;
    *)
      run_combined_service_command
      ;;
  esac
}

run_reset() {
  if ask_yes_no "确认清空 .director-install 安装状态？不会删除 ComfyUI、数据库或模型" n; then
    rm -rf "$STATE_DIR"
    ok "已清空安装状态"
  else
    warn "已取消"
  fi
}

main() {
  parse_args "$@"

  case "$COMMAND" in
    install)
      run_install
      ;;
    check)
      run_check
      ;;
    verify)
      run_verify
      ;;
    reset)
      run_reset
      ;;
    start|stop|restart|status|logs|start-director|stop-director|restart-director|status-director|logs-director|start-comfyui|stop-comfyui|restart-comfyui|status-comfyui|logs-comfyui)
      run_service_command
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage >&2
      exit 64
      ;;
  esac
}

main "$@"
