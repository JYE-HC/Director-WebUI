#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
DATA_DIR="$SCRIPT_DIR/data"
BACKEND_UNIT="director-backend.service"
FRONTEND_UNIT="director-frontend.service"
NODE_BIN_DIR="${DIRECTOR_NODE_BIN_DIR:-}"

BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8787"
FRONTEND_HOST="127.0.0.1"
FRONTEND_PORT="4173"

if [[ -f "$SCRIPT_DIR/.director-install/env.sh" ]]; then
  declare -A _external_env
  declare _var
  for _var in DIRECTOR_HOST DIRECTOR_PORT DIRECTOR_FRONTEND_HOST DIRECTOR_FRONTEND_PORT DIRECTOR_API_ORIGIN DIRECTOR_NODE_BIN_DIR DIRECTOR_COMFYUI_ROOT DIRECTOR_COMFYUI_PORT DIRECTOR_COMFYUI_LISTEN DIRECTOR_COMFYUI_URL DIRECTOR_TMPDIR DIRECTOR_FORCE_SUPERVISOR; do
    [[ -v $_var ]] && _external_env[$_var]="${!_var}"
  done
  # shellcheck disable=SC1090
  source "$SCRIPT_DIR/.director-install/env.sh"
  for _var in "${!_external_env[@]}"; do
    printf -v "$_var" '%s' "${_external_env[$_var]}"
    export "$_var"
  done
fi
BACKEND_HOST="${DIRECTOR_HOST:-$BACKEND_HOST}"
BACKEND_PORT="${DIRECTOR_PORT:-$BACKEND_PORT}"
FRONTEND_HOST="${DIRECTOR_FRONTEND_HOST:-$FRONTEND_HOST}"
FRONTEND_PORT="${DIRECTOR_FRONTEND_PORT:-$FRONTEND_PORT}"

usage() {
  cat <<'EOF'
用法：
  ./director.sh start [选项]
  ./director.sh restart [选项]
  ./director.sh stop
  ./director.sh status
  ./director.sh logs [backend|frontend]

启动选项：
  --host HOST             同时设置前后端监听地址
  --backend-host HOST     设置后端监听地址（默认 127.0.0.1）
  --backend-port PORT     设置后端端口（默认 8787）
  --frontend-host HOST    设置前端监听地址（默认 127.0.0.1）
  --frontend-port PORT    设置前端端口（默认 4173）
  -h, --help              显示帮助

示例：
  ./director.sh start
  ./director.sh restart --host 0.0.0.0 --backend-port 8788 --frontend-port 4174
EOF
}

systemd_available() {
  command -v systemctl >/dev/null 2>&1 || return 1
  command -v systemd-run >/dev/null 2>&1 || return 1
  [[ -d /run/systemd/system ]] || return 1
  timeout 3 systemctl --user show-environment >/dev/null 2>&1
}

die() {
  printf '错误：%s\n' "$*" >&2
  exit 1
}

resolve_node_bin_dir() {
  local node_path
  if [[ -z "$NODE_BIN_DIR" ]]; then
    node_path="$(command -v node 2>/dev/null || true)"
    [[ -n "$node_path" ]] || \
      die "找不到 node；请安装 Node.js 22.13+，或设置 DIRECTOR_NODE_BIN_DIR"
    NODE_BIN_DIR="$(cd -- "$(dirname -- "$node_path")" && pwd)"
  fi
  [[ -x "$NODE_BIN_DIR/node" ]] || \
    die "找不到 node：$NODE_BIN_DIR/node（可通过 DIRECTOR_NODE_BIN_DIR 覆盖）"
  [[ -x "$NODE_BIN_DIR/npm" ]] || \
    die "找不到 npm：$NODE_BIN_DIR/npm（可通过 DIRECTOR_NODE_BIN_DIR 覆盖）"
}

validate_host() {
  local value="$1"
  [[ "$value" =~ ^[A-Za-z0-9._:-]+$ ]] || die "无效的监听地址：$value"
}

validate_port() {
  local value="$1"
  [[ "$value" =~ ^[0-9]+$ ]] || die "端口必须是数字：$value"
  ((10#$value >= 1 && 10#$value <= 65535)) || die "端口超出范围：$value"
}

parse_start_options() {
  while (($#)); do
    case "$1" in
      --host)
        (($# >= 2)) || die "--host 缺少参数"
        BACKEND_HOST="$2"
        FRONTEND_HOST="$2"
        shift 2
        ;;
      --backend-host)
        (($# >= 2)) || die "--backend-host 缺少参数"
        BACKEND_HOST="$2"
        shift 2
        ;;
      --backend-port)
        (($# >= 2)) || die "--backend-port 缺少参数"
        BACKEND_PORT="$2"
        shift 2
        ;;
      --frontend-host)
        (($# >= 2)) || die "--frontend-host 缺少参数"
        FRONTEND_HOST="$2"
        shift 2
        ;;
      --frontend-port)
        (($# >= 2)) || die "--frontend-port 缺少参数"
        FRONTEND_PORT="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *) die "未知选项：$1" ;;
    esac
  done

  validate_host "$BACKEND_HOST"
  validate_host "$FRONTEND_HOST"
  validate_port "$BACKEND_PORT"
  validate_port "$FRONTEND_PORT"
  [[ "$BACKEND_PORT" != "$FRONTEND_PORT" ]] || die "前后端不能使用同一个端口"
}

unit_is_loaded() {
  [[ "$(systemctl --user show "$1" --property=LoadState --value 2>/dev/null || true)" == "loaded" ]]
}

unit_is_active() {
  systemctl --user is-active --quiet "$1"
}

wait_until_unloaded() {
  local unit="$1"
  local attempt
  for attempt in {1..50}; do
    unit_is_loaded "$unit" || return 0
    sleep 0.1
  done
  die "服务停止后仍未卸载：$unit"
}

stop_units() {
  local found=false
  local unit
  for unit in "$FRONTEND_UNIT" "$BACKEND_UNIT"; do
    if unit_is_loaded "$unit"; then
      found=true
      systemctl --user stop "$unit"
      systemctl --user reset-failed "$unit" 2>/dev/null || true
      wait_until_unloaded "$unit"
    fi
  done

  if [[ "$found" == true ]]; then
    printf 'Director 前后端已关闭。\n'
  else
    printf 'Director 前后端未运行。\n'
  fi
}

start_units() {
  resolve_node_bin_dir
  [[ -x "$SCRIPT_DIR/.venv/bin/director-web" ]] || \
    die "后端入口不存在，请先执行 ./install.sh install --comfyui-root /path/to/ComfyUI"
  [[ -d "$FRONTEND_DIR/node_modules" ]] || \
    die "前端依赖不存在，请先在 frontend 目录执行 npm ci"

  if unit_is_loaded "$BACKEND_UNIT" || unit_is_loaded "$FRONTEND_UNIT"; then
    die "Director 服务已经存在；请使用 restart，或先执行 stop"
  fi

  mkdir -p "$DATA_DIR"

  systemd-run --user --collect \
    --unit="$BACKEND_UNIT" \
    --description="Director Web backend" \
    --working-directory="$SCRIPT_DIR" \
    --property="Restart=on-failure" \
    --property="RestartSec=2" \
    --property="StandardOutput=append:$DATA_DIR/director-backend.log" \
    --property="StandardError=append:$DATA_DIR/director-backend.log" \
    --setenv="UV_CACHE_DIR=/tmp/director-web-uv-cache" \
    --setenv="DIRECTOR_HOST=$BACKEND_HOST" \
    --setenv="DIRECTOR_PORT=$BACKEND_PORT" \
    "$SCRIPT_DIR/.venv/bin/director-web" >/dev/null

  if ! systemd-run --user --collect \
    --unit="$FRONTEND_UNIT" \
    --description="Director Web frontend" \
    --working-directory="$FRONTEND_DIR" \
    --property="Restart=on-failure" \
    --property="RestartSec=2" \
    --property="StandardOutput=append:$DATA_DIR/director-frontend.log" \
    --property="StandardError=append:$DATA_DIR/director-frontend.log" \
    --setenv="PATH=$NODE_BIN_DIR:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --setenv="DIRECTOR_FRONTEND_HOST=$FRONTEND_HOST" \
    --setenv="DIRECTOR_FRONTEND_PORT=$FRONTEND_PORT" \
    --setenv="DIRECTOR_API_ORIGIN=http://127.0.0.1:$BACKEND_PORT" \
    "$NODE_BIN_DIR/npm" run dev >/dev/null; then
    systemctl --user stop "$BACKEND_UNIT" || true
    die "前端服务启动失败，后端已回滚关闭"
  fi

  sleep 0.5
  unit_is_active "$BACKEND_UNIT" || die "后端未能保持运行，请查看 $DATA_DIR/director-backend.log"
  unit_is_active "$FRONTEND_UNIT" || die "前端未能保持运行，请查看 $DATA_DIR/director-frontend.log"

  printf 'Director 已启动：\n'
  printf '  前端：http://%s:%s\n' "$FRONTEND_HOST" "$FRONTEND_PORT"
  printf '  后端：http://%s:%s\n' "$BACKEND_HOST" "$BACKEND_PORT"
  if [[ "$FRONTEND_HOST" == "0.0.0.0" || "$BACKEND_HOST" == "0.0.0.0" ]]; then
    printf '提示：0.0.0.0 是监听地址；访问时请使用服务器的实际 IP。\n'
  fi
}

show_status() {
  systemctl --user --no-pager --full status "$BACKEND_UNIT" "$FRONTEND_UNIT" || true
}

show_logs() {
  local target="${1:-}"
  case "$target" in
    "") journalctl --user -u "$BACKEND_UNIT" -u "$FRONTEND_UNIT" -f ;;
    backend) journalctl --user -u "$BACKEND_UNIT" -f ;;
    frontend) journalctl --user -u "$FRONTEND_UNIT" -f ;;
    *) die "logs 只接受 backend 或 frontend" ;;
  esac
}

main() {
  local command="${1:-}"
  [[ -n "$command" ]] || {
    usage
    exit 1
  }
  shift

  if [[ "${DIRECTOR_FORCE_SUPERVISOR:-false}" == true ]] || ! systemd_available; then
    [[ -x "$SCRIPT_DIR/.venv/bin/python" ]] || \
      die "内置 supervisor 需要后端 Python；请先运行 ./bootstrap.sh install"
    exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/tools/director_supervisor.py" "$command" "$@"
  fi

  case "$command" in
    start)
      parse_start_options "$@"
      start_units
      ;;
    restart)
      parse_start_options "$@"
      stop_units
      start_units
      ;;
    stop)
      (($# == 0)) || die "stop 不接受额外参数"
      stop_units
      ;;
    status)
      (($# == 0)) || die "status 不接受额外参数"
      show_status
      ;;
    logs)
      (($# <= 1)) || die "logs 最多接受一个参数"
      show_logs "${1:-}"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      die "未知命令：$command（使用 --help 查看帮助）"
      ;;
  esac
}

main "$@"
