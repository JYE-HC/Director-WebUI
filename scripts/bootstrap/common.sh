#!/usr/bin/env bash
# shellcheck shell=bash
# Common helpers for the Director Web bootstrap installer.

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Global paths are initialized by bootstrap.sh before sourcing this file:
#   SCRIPT_DIR, STATE_DIR, STATE_FILE, ENV_FILE, SETUP_LOG, LOCK_DIR,
#   TOOLS_DIR, TOOLS_BIN_DIR
# ---------------------------------------------------------------------------

STATE_VERSION=1
STATE_READONLY=false

# UI -------------------------------------------------------------------------

if [[ -t 1 ]]; then
  C_BOLD=$'\033[1m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_RED=$'\033[31m'
  C_RESET=$'\033[0m'
else
  C_BOLD=""
  C_GREEN=""
  C_YELLOW=""
  C_RED=""
  C_RESET=""
fi

info()  { printf '%s\n' "$*"; }
ok()    { printf '%s[PASS]%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn()  { printf '%s[WARN]%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
error() { printf '%s[FAIL]%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; }
section() {
  printf '\n%s=================================================================%s\n' "$C_BOLD" "$C_RESET"
  printf '%s%s%s\n' "$C_BOLD" "$*" "$C_RESET"
  printf '%s=================================================================%s\n' "$C_BOLD" "$C_RESET"
}

die() {
  error "$*"
  exit "${2:-1}"
}

# State ----------------------------------------------------------------------

load_state() {
  [[ -f "$STATE_FILE" ]] || return 0
  # shellcheck disable=SC1090
  source "$STATE_FILE"
}

state_set() {
  [[ "$STATE_READONLY" == false ]] || return 0
  local key="$1" value="$2"
  mkdir -p "$STATE_DIR"
  printf '%s=%q\n' "$key" "$value" >>"$STATE_FILE"
}

state_flush() {
  return 0
}

state_get() {
  local key="$1" fallback="${2:-}"
  printf '%s\n' "${!key:-$fallback}"
}

step_status() {
  local step="$1" fallback="${2:-}"
  printf '%s\n' "${!step:-$fallback}"
}

step_is_done() {
  [[ "$(step_status "STEP_$1")" == "done" ]]
}

step_is_skipped() {
  [[ " ${SKIP_STEPS[*]:-} " == *" $1 "* ]]
}

mark_step() {
  state_set "STEP_$1" "$2"
  state_flush
}

# Interactive helpers ---------------------------------------------------------

ask_yes_no() {
  local prompt="$1" default="${2:-n}"
  local answer suffix
  [[ "$ASSUME_YES" == true ]] && {
    [[ "$default" == y ]] && { info "$prompt [Y/n]: Y"; return 0; }
    info "$prompt [y/N]: N"
    return 1
  }
  if [[ "$default" == y ]]; then suffix="[Y/n]"; else suffix="[y/N]"; fi
  while true; do
    printf '%s %s ' "$prompt" "$suffix"
    read -r answer
    answer="${answer,,}"
    [[ -z "$answer" ]] && answer="$default"
    case "$answer" in
      y|yes) return 0 ;;
      n|no) return 1 ;;
      *) warn "请输入 y 或 n" ;;
    esac
  done
}

ask_choice() {
  local title="$1" default="$2" line
  shift 2
  if [[ "$ASSUME_YES" == true ]]; then
    printf '%s\n' "$title" >&2
    local index=1
    for line in "$@"; do printf '  %s) %s\n' "$index" "$line" >&2; index=$((index + 1)); done
    printf '  使用默认选择：%s\n' "$default" >&2
    printf '%s\n' "$default"
    return 0
  fi
  local index
  while true; do
    printf '%s\n' "$title" >&2
    local i=1
    for line in "$@"; do printf '  %s) %s\n' "$i" "$line" >&2; i=$((i + 1)); done
    printf '请选择 [%s]: ' "$default" >&2
    read -r index
    [[ -z "$index" ]] && index="$default"
    if [[ "$index" =~ ^[0-9]+$ ]] && ((index >= 1 && index <= $#)); then
      printf '%s\n' "$index"
      return 0
    fi
    warn "请输入 1-$#"
  done
}

ask_text() {
  local prompt="$1" default="$2"
  local value
  if [[ "$ASSUME_YES" == true ]]; then
    [[ -n "$default" ]] && { printf '%s [%s]\n' "$prompt" "$default" >&2; printf '%s\n' "$default"; return 0; }
    die "$prompt（非交互模式缺少 --yes 默认值）"
  fi
  if [[ -n "$default" ]]; then
    printf '%s [%s]: ' "$prompt" "$default" >&2
  else
    printf '%s: ' "$prompt" >&2
  fi
  read -r value
  printf '%s\n' "${value:-$default}"
}

# Step runner -----------------------------------------------------------------

SKIP_STEPS=()
ONLY_STEP=""
FROM_STEP=""

add_skip_steps() {
  local raw="$1" item
  IFS=, read -r -a raw_items <<<"$raw" || true
  for item in "${raw_items[@]:-}"; do
    [[ -n "$item" ]] && SKIP_STEPS+=("$item")
  done
}

step_is_active() {
  local step="$1"
  [[ -z "$ONLY_STEP" || "$step" == "$ONLY_STEP" ]] || return 1
  [[ -z "$FROM_STEP" || "$FROM_ACTIVE" == true || "$step" == "$FROM_STEP" ]] || return 1
  return 0
}

FROM_ACTIVE=false

run_chain() {
  local step
  for step in "${STEP_QUEUE[@]}"; do
    if [[ "$step" == "$FROM_STEP" ]]; then FROM_ACTIVE=true; fi
    if ! step_is_active "$step"; then continue; fi
    if step_is_skipped "$step"; then
      warn "跳过步骤：$step"
      continue
    fi
    if step_is_done "$step" && [[ -z "$ONLY_STEP" ]]; then
      ok "步骤已完成，跳过：$step"
      continue
    fi

    section "[$step] ${STEP_TITLES[$step]:-$step}"
    if run_step "$step"; then
      mark_step "$step" done
      ok "步骤完成：$step"
    else
      mark_step "$step" failed
      error "安装中断在步骤：$step"
      error "修复后可执行 ./bootstrap.sh --resume 继续；或 ./bootstrap.sh --only $step"
      return 1
    fi
  done
}

run_step() {
  local step="$1"
  case "$step" in
    detect_platform) step_detect_platform ;;
    resolve_layout) step_resolve_layout ;;
    select_comfyui_mode) step_select_comfyui_mode ;;
    install_system_packages) step_install_system_packages ;;
    install_uv) step_install_uv ;;
    install_node) step_install_node ;;
    install_ffmpeg) step_install_ffmpeg ;;
    prepare_comfyui) step_prepare_comfyui ;;
    install_comfyui_deps) step_install_comfyui_deps ;;
    install_director) step_install_director ;;
    verify_offline) step_verify_offline ;;
    verify_online_optional) step_verify_online_optional ;;
    configure_launcher) step_configure_launcher ;;
    start_services_optional) step_start_services_optional ;;
    *) error "未知步骤：$step"; return 1 ;;
  esac
}

# Lock ------------------------------------------------------------------------

acquire_lock() {
  if ! mkdir -p "$LOCK_DIR" 2>/dev/null; then
    error "无法创建安装锁目录：$LOCK_DIR"
    return 1
  fi
  local pid_file="$LOCK_DIR/pid"
  if [[ -f "$pid_file" ]]; then
    local previous_pid
    previous_pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$previous_pid" && -d "/proc/$previous_pid" ]]; then
      error "另一个安装进程正在运行（PID $previous_pid）"
      return 1
    fi
    warn "清除过期安装锁（PID ${previous_pid:-unknown}）"
    rm -rf "$LOCK_DIR"
    mkdir -p "$LOCK_DIR" || return 1
  fi
  printf '%s\n' "$$" >"$pid_file"
  trap 'release_lock' EXIT
}

release_lock() {
  [[ -f "$LOCK_DIR/pid" ]] || return 0
  local current_pid
  current_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [[ "$current_pid" == "$$" ]]; then
    rm -rf "$LOCK_DIR" || true
  fi
}
