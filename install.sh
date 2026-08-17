#!/usr/bin/env bash

set -Eeuo pipefail
umask 022

readonly EX_OK=0
readonly EX_INCOMPATIBLE=2
readonly EX_SAFETY=3
readonly EX_INSTALL=4
readonly EX_VERIFY=5
readonly EX_USAGE=64

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly TESTED_COMFY_COMMIT="8f37cf8c833a8f2d3c62e2adbccebfd165623481"
readonly NODE_NAMES=("raylight" "ComfyUI-MiniMax-H3-Turbo")

COMMAND=""
COMFYUI_ROOT="${COMFYUI_ROOT:-}"
COMFYUI_PYTHON="${COMFYUI_PYTHON:-}"
NODE_BIN_DIR="${DIRECTOR_NODE_BIN_DIR:-}"
COMFY_URL=""
CONFIRM_STOPPED=false
ASSUME_YES=false
DRY_RUN=false
REQUIRE_RAYLIGHT=false
COMFY_DIRTY=false
declare -A REPLACE_NODES=()
declare -A NODE_STATUS=()

usage() {
  cat <<'EOF'
用法：
  ./install.sh check   --comfyui-root PATH [选项]
  ./install.sh install --comfyui-root PATH [选项]
  ./install.sh verify  --comfyui-root PATH [选项]

通用选项：
  --comfyui-root PATH       ComfyUI checkout 根目录（也可设置 COMFYUI_ROOT）
  --comfy-python PATH       ComfyUI Python；默认 PATH/.venv/bin/python
  --node-bin-dir PATH       node 与 npm 所在目录
  --require-raylight        将少于两张可用 CK GPU 视为失败
  --comfy-url URL           verify 时额外检查在线 ComfyUI API

install 安全选项：
  --replace-node NAME       备份并替换冲突节点；可重复，NAME 只能是两个 bundled 节点
  --confirm-comfyui-stopped 确认替换节点前已停止 ComfyUI
  --dry-run                 只检查并显示计划，不写入
  -y, --yes                 跳过安装确认

安装器不会 pull/checkout/patch ComfyUI，不下载模型，也不会停止或重启 ComfyUI。
EOF
}

die_usage() {
  printf '[FAIL] %s\n' "$*" >&2
  usage >&2
  exit "$EX_USAGE"
}

pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*" >&2; }

parse_args() {
  (($# >= 1)) || die_usage "缺少 check、install 或 verify"
  COMMAND="$1"
  shift
  case "$COMMAND" in
    check|install|verify) ;;
    -h|--help|help) usage; exit "$EX_OK" ;;
    *) die_usage "未知命令：$COMMAND" ;;
  esac

  while (($#)); do
    case "$1" in
      --comfyui-root|--comfyui)
        (($# >= 2)) || die_usage "$1 缺少路径"
        COMFYUI_ROOT="$2"
        shift 2
        ;;
      --comfy-python)
        (($# >= 2)) || die_usage "$1 缺少路径"
        COMFYUI_PYTHON="$2"
        shift 2
        ;;
      --node-bin-dir)
        (($# >= 2)) || die_usage "$1 缺少路径"
        NODE_BIN_DIR="$2"
        shift 2
        ;;
      --comfy-url)
        (($# >= 2)) || die_usage "$1 缺少 URL"
        COMFY_URL="$2"
        shift 2
        ;;
      --confirm-comfyui-stopped) CONFIRM_STOPPED=true; shift ;;
      --require-raylight) REQUIRE_RAYLIGHT=true; shift ;;
      --dry-run) DRY_RUN=true; shift ;;
      -y|--yes) ASSUME_YES=true; shift ;;
      --replace-node)
        (($# >= 2)) || die_usage "$1 缺少节点名"
        case "$2" in
          raylight|ComfyUI-MiniMax-H3-Turbo) REPLACE_NODES["$2"]=true ;;
          *) die_usage "不能替换未知节点：$2" ;;
        esac
        shift 2
        ;;
      -h|--help) usage; exit "$EX_OK" ;;
      *) die_usage "未知选项：$1" ;;
    esac
  done
  [[ -n "$COMFYUI_ROOT" ]] || die_usage "必须传入 --comfyui-root PATH"
  [[ "$COMMAND" == "install" ]] || {
    [[ "$CONFIRM_STOPPED" == false && "$DRY_RUN" == false && ${#REPLACE_NODES[@]} -eq 0 ]] || \
      die_usage "安装安全选项只能与 install 一起使用"
  }
}

resolve_paths() {
  [[ -d "$COMFYUI_ROOT" ]] || {
    fail "ComfyUI 目录不存在：$COMFYUI_ROOT"
    return "$EX_INCOMPATIBLE"
  }
  COMFYUI_ROOT="$(cd -- "$COMFYUI_ROOT" && pwd -P)"
  if [[ -z "$COMFYUI_PYTHON" ]]; then
    COMFYUI_PYTHON="$COMFYUI_ROOT/.venv/bin/python"
  fi
  [[ -x "$COMFYUI_PYTHON" ]] || {
    fail "ComfyUI Python 不可执行：$COMFYUI_PYTHON"
    return "$EX_INCOMPATIBLE"
  }
  if [[ -z "$NODE_BIN_DIR" ]]; then
    local node_path
    node_path="$(command -v node 2>/dev/null || true)"
    [[ -n "$node_path" ]] || {
      fail "找不到 node；请传入 --node-bin-dir"
      return "$EX_INCOMPATIBLE"
    }
    NODE_BIN_DIR="$(cd -- "$(dirname -- "$node_path")" && pwd -P)"
  else
    [[ -d "$NODE_BIN_DIR" ]] || {
      fail "Node bin 目录不存在：$NODE_BIN_DIR"
      return "$EX_INCOMPATIBLE"
    }
    NODE_BIN_DIR="$(cd -- "$NODE_BIN_DIR" && pwd -P)"
  fi
}

check_command() {
  command -v "$1" >/dev/null 2>&1 || {
    fail "缺少命令：$1"
    return 1
  }
  pass "命令可用：$1"
}

check_node() {
  [[ -x "$NODE_BIN_DIR/node" && -x "$NODE_BIN_DIR/npm" ]] || {
    fail "需要同一目录中的 node 和 npm：$NODE_BIN_DIR"
    return 1
  }
  local version major minor
  version="$(PATH="$NODE_BIN_DIR:$PATH" "$NODE_BIN_DIR/node" -p 'process.versions.node')"
  IFS=. read -r major minor _ <<<"$version"
  if ((major == 22 && minor >= 13)) || ((major >= 24)); then
    pass "Node.js $version"
  else
    fail "Node.js $version 不满足 ^22.13 或 >=24"
    return 1
  fi
  pass "npm $(PATH="$NODE_BIN_DIR:$PATH" "$NODE_BIN_DIR/npm" --version)"
}

check_ffmpeg_features() {
  local encoders filters
  encoders="$(ffmpeg -hide_banner -encoders 2>&1)"
  filters="$(ffmpeg -hide_banner -filters 2>&1)"
  for encoder in libx264 aac; do
    if grep -Eq "[[:space:]]${encoder}[[:space:]]" <<<"$encoders"; then
      pass "ffmpeg encoder：$encoder"
    else
      fail "ffmpeg 缺少 encoder：$encoder"
      return 1
    fi
  done
  for filter in fps scale select aresample; do
    if grep -Eq "[[:space:]]${filter}[[:space:]]" <<<"$filters"; then
      pass "ffmpeg filter：$filter"
    else
      fail "ffmpeg 缺少 filter：$filter"
      return 1
    fi
  done
}

check_comfy_tree() {
  local path
  for path in main.py comfy custom_nodes; do
    [[ -e "$COMFYUI_ROOT/$path" ]] || {
      fail "ComfyUI 目录缺少 $path"
      return 1
    }
  done
  [[ -d "$COMFYUI_ROOT/.git" ]] || {
    fail "ComfyUI 必须是 Git checkout，安装器不会猜测 portable 包的版本"
    return 1
  }
  if ! git -C "$COMFYUI_ROOT" cat-file -e "${TESTED_COMFY_COMMIT}^{commit}" 2>/dev/null; then
    fail "ComfyUI checkout 不含兼容基线；若是更新的 shallow clone，请手动加深后重试"
    return 1
  fi
  if ! git -C "$COMFYUI_ROOT" merge-base --is-ancestor "$TESTED_COMFY_COMMIT" HEAD; then
    fail "ComfyUI 必须与实测基线相同，或是它的官方后继版本：$TESTED_COMFY_COMMIT"
    return 1
  fi
  local head tracked untracked
  head="$(git -C "$COMFYUI_ROOT" rev-parse HEAD)"
  if [[ "$head" == "$TESTED_COMFY_COMMIT" ]]; then
    pass "ComfyUI 位于实测 commit ${head:0:12}"
  else
    pass "ComfyUI ${head:0:12} 是实测基线 ${TESTED_COMFY_COMMIT:0:12} 的后继版本"
    warn "该更新版本尚未完成本发布的全套 GPU 冒烟测试；安装后会继续检查实时节点/API 契约"
  fi
  tracked="$(git -C "$COMFYUI_ROOT" status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
  untracked="$(git -C "$COMFYUI_ROOT" ls-files --others --exclude-standard | wc -l | tr -d ' ')"
  if ((tracked > 0 || untracked > 0)); then
    COMFY_DIRTY=true
    warn "ComfyUI 工作树有本地状态（tracked=$tracked, untracked=$untracked）；不显示文件名以避免泄露"
  else
    pass "ComfyUI 工作树干净"
  fi
}

is_replace_allowed() {
  [[ "${REPLACE_NODES[$1]:-false}" == true ]]
}

check_node_collisions() {
  local name target status safety=0
  for name in "${NODE_NAMES[@]}"; do
    target="$COMFYUI_ROOT/custom_nodes/$name"
    status="$($COMFYUI_PYTHON "$SCRIPT_DIR/tools/release_installer.py" node-status "$name" "$target")"
    NODE_STATUS["$name"]="$status"
    case "$status" in
      absent) pass "$name：尚未安装" ;;
      same) pass "$name：已是本发布内容（无需改动）" ;;
      symlink)
        fail "$name：目标是符号链接，安装器拒绝修改"
        safety=1
        ;;
      conflict)
        if is_replace_allowed "$name"; then
          if [[ "$CONFIRM_STOPPED" == true ]]; then
            warn "$name：将备份并替换已有不同内容"
          else
            fail "$name：替换前必须加 --confirm-comfyui-stopped"
            safety=1
          fi
        else
          fail "$name：已有内容不同；默认不覆盖。需要时显式传入 --replace-node $name"
          safety=1
        fi
        ;;
      *) fail "$name：无法判定目标状态"; safety=1 ;;
    esac
  done
  return "$safety"
}

run_readonly_checks() {
  local phase="$1" incompatible=0 safety=0
  [[ "$(uname -s)" == Linux ]] || { fail "当前只支持 Linux"; incompatible=1; }
  for command in git uv ffmpeg ffprobe; do
    check_command "$command" || incompatible=1
  done
  if command -v systemd-run >/dev/null 2>&1 && command -v systemctl >/dev/null 2>&1; then
    pass "systemd 用户服务工具可用"
    systemctl --user show-environment >/dev/null 2>&1 || \
      warn "当前 shell 无法访问 systemd user bus；登录会话中启动 Director 时再确认"
  else
    warn "缺少 systemd-run/systemctl；director.sh 不可用，但可手动启动前后端"
  fi
  check_node || incompatible=1
  check_ffmpeg_features || incompatible=1
  check_comfy_tree || incompatible=1
  "$COMFYUI_PYTHON" "$SCRIPT_DIR/tools/release_installer.py" payload-check || incompatible=1
  local ray_args=()
  [[ "$REQUIRE_RAYLIGHT" == true ]] && ray_args+=(--require-raylight)
  "$COMFYUI_PYTHON" "$SCRIPT_DIR/tools/check_comfy_environment.py" \
    --phase "$phase" "${ray_args[@]}" || incompatible=1
  check_node_collisions || safety=1
  ((incompatible == 0)) || return "$EX_INCOMPATIBLE"
  ((safety == 0)) || return "$EX_SAFETY"
  return "$EX_OK"
}

confirm_install() {
  [[ "$ASSUME_YES" == true ]] && return 0
  printf '\n即将安装 Director 依赖、构建前端，并把 bundled 节点发布到：\n  %s/custom_nodes\n' "$COMFYUI_ROOT"
  printf '不会修改 ComfyUI core，也不会重启 ComfyUI。继续？ [y/N] '
  local answer
  read -r answer
  [[ "$answer" == y || "$answer" == Y || "$answer" == yes || "$answer" == YES ]]
}

install_dependencies() {
  printf '\n== 安装 Director 后端 ==\n'
  (cd -- "$SCRIPT_DIR" && UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/director-web-uv-cache}" uv sync --frozen --no-dev)
  printf '\n== 安装并构建前端 ==\n'
  (cd -- "$SCRIPT_DIR/frontend" && PATH="$NODE_BIN_DIR:$PATH" "$NODE_BIN_DIR/npm" ci --no-audit --no-fund)
  (cd -- "$SCRIPT_DIR/frontend" && PATH="$NODE_BIN_DIR:$PATH" "$NODE_BIN_DIR/npm" run build)

  printf '\n== 安装 RayLight Python 依赖到 ComfyUI venv ==\n'
  local constraint torch_version
  constraint="$(mktemp)"
  torch_version="$($COMFYUI_PYTHON -c 'import importlib.metadata; print(importlib.metadata.version("torch"))')"
  printf 'torch==%s\n' "$torch_version" >"$constraint"
  if ! UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/director-web-uv-cache}" uv pip install \
    --python "$COMFYUI_PYTHON" \
    --constraint "$constraint" \
    --requirements "$SCRIPT_DIR/custom_nodes/raylight/requirements.txt"; then
    rm -f -- "$constraint"
    return 1
  fi
  rm -f -- "$constraint"
}

verify_registry_paths() {
  local raylight_root="$1" turbo_root="$2"
  "$COMFYUI_PYTHON" "$SCRIPT_DIR/tools/verify_comfy_registry.py" \
    --comfyui-root "$COMFYUI_ROOT" \
    --raylight-root "$raylight_root" \
    --turbo-root "$turbo_root"
}

STAGE_BASE=""
ROLLBACK_NEEDED=false
declare -a INSTALLED_NAMES=()
declare -A BACKUP_PATHS=()

cleanup_stage() {
  if [[ -n "$STAGE_BASE" && -d "$STAGE_BASE" ]]; then
    rm -rf -- "$STAGE_BASE"
  fi
}

rollback_nodes() {
  [[ "$ROLLBACK_NEEDED" == true ]] || return 0
  warn "节点发布未完成，正在回滚"
  local index name target backup
  for ((index=${#INSTALLED_NAMES[@]}-1; index>=0; index--)); do
    name="${INSTALLED_NAMES[$index]}"
    target="$COMFYUI_ROOT/custom_nodes/$name"
    backup="${BACKUP_PATHS[$name]:-}"
    if [[ -e "$target" && ! -L "$target" ]]; then
      mv -- "$target" "$STAGE_BASE/rollback-$name" || true
    fi
    if [[ -n "$backup" && -d "$backup" ]]; then
      mv -- "$backup" "$target" || true
    fi
  done
  ROLLBACK_NEEDED=false
}

publish_nodes() {
  local name source staged expected actual target backup_root backup timestamp old_digest
  STAGE_BASE="$(mktemp -d "$COMFYUI_ROOT/custom_nodes/.director-stage.XXXXXX")"
  trap 'rollback_nodes; cleanup_stage' EXIT

  for name in "${NODE_NAMES[@]}"; do
    [[ "${NODE_STATUS[$name]}" == same ]] && continue
    source="$SCRIPT_DIR/custom_nodes/$name"
    staged="$STAGE_BASE/$name"
    mkdir -- "$staged"
    cp -a -- "$source/." "$staged/"
    expected="$($COMFYUI_PYTHON "$SCRIPT_DIR/tools/release_installer.py" tree-digest "$source")"
    actual="$($COMFYUI_PYTHON "$SCRIPT_DIR/tools/release_installer.py" tree-digest "$staged")"
    [[ "$actual" == "$expected" ]] || { fail "$name staging digest 不一致"; return 1; }
  done

  backup_root="$COMFYUI_ROOT/.director-backups/custom_nodes"
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  ROLLBACK_NEEDED=true
  for name in "${NODE_NAMES[@]}"; do
    [[ "${NODE_STATUS[$name]}" == same ]] && continue
    target="$COMFYUI_ROOT/custom_nodes/$name"
    backup=""
    if [[ "${NODE_STATUS[$name]}" == conflict ]]; then
      mkdir -p -- "$backup_root"
      old_digest="$($COMFYUI_PYTHON "$SCRIPT_DIR/tools/release_installer.py" tree-digest "$target")"
      backup="$backup_root/${name}-${timestamp}-${old_digest:0:12}"
      [[ ! -e "$backup" ]] || { fail "备份路径已存在：$backup"; return 1; }
      mv -- "$target" "$backup"
      BACKUP_PATHS["$name"]="$backup"
      printf '[PASS] 已备份 %s 到 %s\n' "$name" "$backup"
    fi
    if ! mv -- "$STAGE_BASE/$name" "$target"; then
      fail "发布 $name 失败"
      return 1
    fi
    INSTALLED_NAMES+=("$name")
    pass "已发布 $name"
  done
}

finish_node_publish() {
  ROLLBACK_NEEDED=false
  cleanup_stage
  trap - EXIT
}

run_verify() {
  local incompatible=0
  [[ -x "$SCRIPT_DIR/.venv/bin/director-web" ]] || { fail "Director 后端尚未安装"; incompatible=1; }
  [[ -d "$SCRIPT_DIR/frontend/dist" ]] || { fail "前端构建目录不存在"; incompatible=1; }
  "$SCRIPT_DIR/.venv/bin/python" -c 'import director' || incompatible=1
  local ray_args=()
  [[ "$REQUIRE_RAYLIGHT" == true ]] && ray_args+=(--require-raylight)
  "$COMFYUI_PYTHON" "$SCRIPT_DIR/tools/check_comfy_environment.py" \
    --phase verify "${ray_args[@]}" || incompatible=1
  local name status
  for name in "${NODE_NAMES[@]}"; do
    status="$($COMFYUI_PYTHON "$SCRIPT_DIR/tools/release_installer.py" node-status \
      "$name" "$COMFYUI_ROOT/custom_nodes/$name")"
    if [[ "$status" == same ]]; then
      pass "$name 安装内容校验通过"
    else
      fail "$name 安装内容不匹配（状态：$status）"
      incompatible=1
    fi
  done
  ((incompatible == 0)) || return "$EX_VERIFY"
  verify_registry_paths \
    "$COMFYUI_ROOT/custom_nodes/raylight" \
    "$COMFYUI_ROOT/custom_nodes/ComfyUI-MiniMax-H3-Turbo" || return "$EX_VERIFY"
  if [[ -n "$COMFY_URL" ]]; then
    "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/tools/check_comfy_online.py" \
      --url "$COMFY_URL" || return "$EX_VERIFY"
  fi
  pass "安装验证完成；未加载模型、未提交 prompt、未启动 Ray cluster"
}

run_install() {
  local check_code=0
  run_readonly_checks preinstall || check_code=$?
  ((check_code == 0)) || return "$check_code"
  if [[ "$DRY_RUN" == true ]]; then
    pass "dry-run 完成：以上计划没有写入任何文件"
    return 0
  fi
  confirm_install || { warn "用户取消安装"; return 130; }

  exec 9>"$COMFYUI_ROOT/.director-install.lock"
  if command -v flock >/dev/null 2>&1; then
    flock -n 9 || { fail "另一个 Director 安装进程正在运行"; return "$EX_SAFETY"; }
  fi

  install_dependencies || { fail "依赖安装或前端构建失败"; return "$EX_INSTALL"; }
  local ray_args=()
  [[ "$REQUIRE_RAYLIGHT" == true ]] && ray_args+=(--require-raylight)
  "$COMFYUI_PYTHON" "$SCRIPT_DIR/tools/check_comfy_environment.py" \
    --phase verify "${ray_args[@]}" || return "$EX_INSTALL"
  verify_registry_paths \
    "$SCRIPT_DIR/custom_nodes/raylight" \
    "$SCRIPT_DIR/custom_nodes/ComfyUI-MiniMax-H3-Turbo" || return "$EX_VERIFY"

  publish_nodes || { rollback_nodes; cleanup_stage; trap - EXIT; return "$EX_INSTALL"; }
  if ! verify_registry_paths \
    "$COMFYUI_ROOT/custom_nodes/raylight" \
    "$COMFYUI_ROOT/custom_nodes/ComfyUI-MiniMax-H3-Turbo"; then
    rollback_nodes
    cleanup_stage
    trap - EXIT
    return "$EX_VERIFY"
  fi
  finish_node_publish
  pass "安装完成。请手动重启 ComfyUI，然后执行："
  printf '  ./install.sh verify --comfyui-root %q\n' "$COMFYUI_ROOT"
  printf '  ./director.sh start\n'
}

main() {
  parse_args "$@"
  resolve_paths || exit "$?"
  case "$COMMAND" in
    check)
      run_readonly_checks preinstall
      ;;
    install)
      run_install
      ;;
    verify)
      "$COMFYUI_PYTHON" "$SCRIPT_DIR/tools/release_installer.py" payload-check
      run_verify
      ;;
  esac
}

main "$@"
