#!/usr/bin/env bash
# shellcheck shell=bash
# ComfyUI bootstrap for Linux and WSL2: existing root reuse, automatic clone,
# venv creation, dependency installation and Windows model sharing.

readonly TESTED_COMFY_COMMIT="8f37cf8c833a8f2d3c62e2adbccebfd165623481"
readonly COMFYUI_DEFAULT_PORT=28188
readonly COMFYUI_DEFAULT_LISTEN="127.0.0.1"

COMFYUI_MODE="auto"          # auto | existing | skip
COMFYUI_ROOT="${COMFYUI_ROOT:-}"
COMFYUI_REF="${COMFYUI_REF:-latest}"
COMFYUI_URL="${COMFYUI_URL:-}"
COMFYUI_PYTHON="${COMFYUI_PYTHON:-}"
COMFYUI_PORT="${COMFYUI_PORT:-$COMFYUI_DEFAULT_PORT}"
COMFYUI_LISTEN="${COMFYUI_LISTEN:-$COMFYUI_DEFAULT_LISTEN}"
COMFYUI_AUTO=false
SKIP_COMFYUI_DEPS=false
WINDOWS_COMFYUI_ROOT=""
WINDOWS_COMFYUI_WSL_ROOT=""
WINDOWS_MODEL_SHARING_ENABLED=false

step_select_comfyui_mode() {
  if [[ -n "${COMFYUI_ROOT_OPTION:-}" ]]; then
    COMFYUI_MODE=existing
    COMFYUI_ROOT="$COMFYUI_ROOT_OPTION"
    ok "使用已有 ComfyUI：$COMFYUI_ROOT"
  elif [[ "${SKIP_COMFYUI:-false}" == true ]]; then
    COMFYUI_MODE=skip
    if [[ -z "$COMFYUI_URL" ]]; then
      COMFYUI_URL="$(ask_text "请输入远程 ComfyUI 地址（例如 http://192.168.1.10:8188）" "")"
    fi
    [[ -n "$COMFYUI_URL" ]] || { error "远程 ComfyUI 地址不能为空"; return 1; }
    ok "跳过本地 ComfyUI，使用远程：$COMFYUI_URL"
  else
    local choice
    choice="$(ask_choice "ComfyUI 安装方式：" 2 \
      "使用已有目录（稍后输入路径）" \
      "自动 clone 官方 ComfyUI 最新 master 到项目目录" \
      "自动 clone 并固定 Director 实测 commit（生产稳定优先）" \
      "跳过本地 ComfyUI，连接远程")" || { error "ComfyUI 选择失败"; return 1; }

    case "$choice" in
      1)
        COMFYUI_MODE=existing
        COMFYUI_ROOT="$(ask_text "请输入 ComfyUI 绝对路径" "${COMFYUI_ROOT:-}")"
        [[ -n "$COMFYUI_ROOT" ]] || { error "ComfyUI 路径不能为空"; return 1; }
        ;;
      2)
        COMFYUI_MODE=auto
        COMFYUI_REF=latest
        COMFYUI_ROOT="$SCRIPT_DIR/ComfyUI"
        ;;
      3)
        COMFYUI_MODE=auto
        COMFYUI_REF=tested
        COMFYUI_ROOT="$SCRIPT_DIR/ComfyUI"
        ;;
      4)
        COMFYUI_MODE=skip
        COMFYUI_URL="$(ask_text "请输入远程 ComfyUI 地址" "")"
        [[ -n "$COMFYUI_URL" ]] || { error "远程 ComfyUI 地址不能为空"; return 1; }
        ;;
      *) error "未知选择：$choice"; return 1 ;;
    esac
    ok "ComfyUI 模式：$COMFYUI_MODE"
  fi

  if [[ "$COMFYUI_MODE" != skip && "$IS_WSL" == true ]]; then
    if [[ -n "$WINDOWS_COMFYUI_ROOT" ]]; then
      WINDOWS_MODEL_SHARING_ENABLED=true
    elif [[ "$ASSUME_YES" == true ]]; then
      WINDOWS_MODEL_SHARING_ENABLED=false
      info "未提供 --windows-comfyui-root，跳过 Windows 模型共享"
    elif ask_yes_no "是否复用 Windows 下 ComfyUI 已下载的模型？（需要提供 Windows 侧 ComfyUI 根目录）" y; then
      WINDOWS_COMFYUI_ROOT="$(ask_text "Windows ComfyUI 绝对路径（如 C:\\path\\to\\ComfyUI）" "")"
      [[ -n "$WINDOWS_COMFYUI_ROOT" ]] && WINDOWS_MODEL_SHARING_ENABLED=true || WINDOWS_MODEL_SHARING_ENABLED=false
    else
      WINDOWS_MODEL_SHARING_ENABLED=false
    fi

    if [[ "$WINDOWS_MODEL_SHARING_ENABLED" == true ]]; then
      WINDOWS_COMFYUI_WSL_ROOT="$(windows_path_to_wsl "$WINDOWS_COMFYUI_ROOT")"
      [[ -n "$WINDOWS_COMFYUI_WSL_ROOT" ]] || { error "Windows 路径转换失败"; return 1; }
      ok "Windows ComfyUI WSL 路径：$WINDOWS_COMFYUI_WSL_ROOT"
    fi
  fi

  state_set COMFYUI_MODE "$COMFYUI_MODE"
  state_set COMFYUI_ROOT "$COMFYUI_ROOT"
  state_set COMFYUI_REF "$COMFYUI_REF"
  state_set COMFYUI_URL "$COMFYUI_URL"
  state_set WINDOWS_MODEL_SHARING_ENABLED "$WINDOWS_MODEL_SHARING_ENABLED"
  state_set WINDOWS_COMFYUI_WSL_ROOT "$WINDOWS_COMFYUI_WSL_ROOT"
  state_flush
  return 0
}

validate_existing_comfyui() {
  local root="$1"
  [[ -d "$root" ]] || { error "ComfyUI 目录不存在：$root"; return 1; }
  [[ -f "$root/main.py" ]] || { error "ComfyUI 目录缺少 main.py：$root"; return 1; }
  [[ -d "$root/comfy" ]] || { error "ComfyUI 目录缺少 comfy/：$root"; return 1; }
  [[ -f "$root/requirements.txt" ]] || { error "ComfyUI 目录缺少 requirements.txt：$root"; return 1; }
  if [[ ! -d "$root/.git" ]]; then
    warn "已有 ComfyUI 不是 Git checkout；现有 install.sh 的版本基线校验会失败"
  fi
  COMFYUI_ROOT="$(cd -- "$root" && pwd -P)"
  ok "ComfyUI 目录有效：$COMFYUI_ROOT"
  return 0
}

clone_or_resume_comfyui() {
  local target="$COMFYUI_ROOT"
  if [[ -e "$target" ]]; then
    if [[ -d "$target/.git" ]]; then
      ok "复用自动安装目录：$target"
    else
      error "目标已存在但不是 Git 仓库：$target"
      error "请手动移除该目录或使用 --comfyui-root 指定其它路径"
      return 1
    fi
  else
    info "git clone https://github.com/comfyanonymous/ComfyUI.git $target"
    git clone https://github.com/comfyanonymous/ComfyUI.git "$target" || {
      error "ComfyUI clone 失败；可执行 ./bootstrap.sh --resume 续传"
      return 1
    }
  fi

  case "$COMFYUI_REF" in
    latest)
      # A fresh clone already sits on the default branch tip. For resume we
      # deliberately avoid surprise updates unless this is our managed copy.
      if [[ "$(git -C "$target" status --porcelain --untracked-files=no | wc -l | tr -d ' ')" == 0 ]]; then
        git -C "$target" fetch --all --prune --tags >/dev/null 2>&1 || warn "ComfyUI fetch 失败，继续使用当前 HEAD"
        local default_branch
        default_branch="$(git -C "$target" symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's#^refs/remotes/origin/##')"
        [[ -n "$default_branch" ]] || default_branch=master
        git -C "$target" checkout "$default_branch" >/dev/null 2>&1 || true
        git -C "$target" merge --ff-only "origin/$default_branch" >/dev/null 2>&1 || warn "ComfyUI 更新失败，继续使用当前 HEAD"
      fi
      ;;
    tested)
      git -C "$target" fetch --all --prune --tags >/dev/null 2>&1 || warn "ComfyUI fetch 失败，尝试本地 checkout"
      git -C "$target" checkout --detach "$TESTED_COMFY_COMMIT" || {
        error "无法 checkout 已测试 ComfyUI commit：$TESTED_COMFY_COMMIT"
        return 1
      }
      ;;
    *)
      git -C "$target" fetch --all --prune --tags >/dev/null 2>&1 || true
      git -C "$target" checkout --detach "$COMFYUI_REF" || {
        error "无法 checkout ComfyUI ref：$COMFYUI_REF"
        return 1
      }
      ;;
  esac

  local head
  head="$(git -C "$target" rev-parse HEAD)"
  info "ComfyUI HEAD：${head:0:12}"
  [[ "$COMFYUI_REF" == latest ]] && {
    if git -C "$target" merge-base --is-ancestor "$TESTED_COMFY_COMMIT" "$head"; then
      ok "ComfyUI 是 Director 兼容基线的后继版本"
    else
      warn "ComfyUI 最新 master 不包含 Director 兼容基线，请使用 --comfyui-ref tested 或已有兼容目录"
      return 1
    fi
  }
  return 0
}

link_windows_models() {
  [[ "$WINDOWS_MODEL_SHARING_ENABLED" == true ]] || return 0
  [[ "$IS_WSL" == true ]] || return 0
  local source_models="$WINDOWS_COMFYUI_WSL_ROOT/models"
  local target_models="$COMFYUI_ROOT/models"
  [[ -d "$source_models" ]] || {
    warn "Windows ComfyUI models 目录不存在：$source_models"
    return 0
  }

  if [[ -L "$target_models" ]]; then
    local existing
    existing="$(readlink -f "$target_models")"
    if [[ "$existing" == "$(readlink -f "$source_models")" ]]; then
      ok "Windows models 已软链接：$target_models -> $source_models"
    else
      warn "WSL ComfyUI models 已是指向其它位置的软链接，保留不动：$existing"
    fi
    return 0
  fi

  if [[ ! -e "$target_models" ]]; then
    mkdir -p "$(dirname "$target_models")"
    ln -s "$source_models" "$target_models"
    ok "已软链接 Windows 模型目录：$target_models -> $source_models"
    return 0
  fi

  if [[ -d "$target_models" ]]; then
    if [[ -z "$(find "$target_models" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]]; then
      rmdir "$target_models" || { error "无法移除空 models 目录"; return 1; }
      ln -s "$source_models" "$target_models"
      ok "已软链接 Windows 模型目录：$target_models -> $source_models"
      return 0
    fi
    warn "WSL ComfyUI models 目录非空，不覆盖；改为通过 extra_model_paths.yaml 追加 Windows 模型路径"
    write_extra_model_paths "$source_models"
    return 0
  fi

  warn "WSL ComfyUI models 不是目录，跳过 Windows 模型共享"
  return 0
}

write_extra_model_paths() {
  local base_path="$1"
  local yaml="$COMFYUI_ROOT/extra_model_paths.yaml"
  if [[ -f "$yaml" ]] && grep -Fq "$base_path" "$yaml"; then
    ok "extra_model_paths.yaml 已包含：$base_path"
    return 0
  fi
  cat >>"$yaml" <<EOF

director_windows_comfyui:
  base_path: $base_path

EOF
  ok "已追加 Windows 模型路径到 $yaml"
}


step_prepare_comfyui() {
  if [[ "$COMFYUI_MODE" == skip ]]; then
    ok "跳过本地 ComfyUI"
    return 0
  fi
  [[ -n "$COMFYUI_ROOT" ]] || { error "ComfyUI 路径为空"; return 1; }

  if [[ "$COMFYUI_MODE" == existing ]]; then
    validate_existing_comfyui "$COMFYUI_ROOT" || return 1
  else
    COMFYUI_ROOT="$(cd -- "$(dirname -- "$COMFYUI_ROOT")" && pwd -P)/$(basename -- "$COMFYUI_ROOT")"
    clone_or_resume_comfyui || return 1
  fi

  link_windows_models || return 1
  COMFYUI_PYTHON="$COMFYUI_ROOT/.venv/bin/python"
  state_set COMFYUI_ROOT "$COMFYUI_ROOT"
  state_set COMFYUI_PYTHON "$COMFYUI_PYTHON"
  state_flush
  ok "ComfyUI 准备完成：$COMFYUI_ROOT"
  return 0
}

step_install_comfyui_deps() {
  if [[ "$COMFYUI_MODE" == skip ]]; then
    ok "跳过 ComfyUI 依赖安装"
    return 0
  fi
  [[ -d "$COMFYUI_ROOT" ]] || { error "ComfyUI 目录不存在"; return 1; }

  local python_bin=""
  if [[ -n "$COMFYUI_PYTHON_OPTION" ]]; then
    python_bin="${COMFYUI_PYTHON_OPTION%/}"
    [[ -x "$python_bin" ]] || { error "指定的 ComfyUI Python 不可执行：$python_bin"; return 1; }
    ok "使用已有 ComfyUI Python：$python_bin"
  elif [[ -x "$COMFYUI_ROOT/.venv/bin/python" ]]; then
    python_bin="$COMFYUI_ROOT/.venv/bin/python"
    ok "复用 ComfyUI venv：$python_bin"
  else
    local uv_python
    uv_python="$("$UV_BIN" python find 3.12 2>/dev/null | head -n 1 || true)"
    if [[ -z "$uv_python" ]]; then
      info "安装 Python 3.12（由 uv 管理）"
      "$UV_BIN" python install 3.12 || { error "uv python install 3.12 失败"; return 1; }
      uv_python="$("$UV_BIN" python find 3.12 2>/dev/null | head -n 1 || true)"
    fi
    [[ -n "$uv_python" ]] || { error "未找到 Python 3.12"; return 1; }
    info "创建 ComfyUI venv：$COMFYUI_ROOT/.venv"
    "$UV_BIN" venv --seed "$COMFYUI_ROOT/.venv" --python "$uv_python" || {
      error "创建 ComfyUI venv 失败"
      return 1
    }
    python_bin="$COMFYUI_ROOT/.venv/bin/python"
  fi

  COMFYUI_PYTHON="$python_bin"
  state_set COMFYUI_PYTHON "$COMFYUI_PYTHON"
  state_flush

  if [[ "$SKIP_COMFYUI_DEPS" == true ]]; then
    warn "已选择跳过 ComfyUI requirements 安装"
    return 0
  fi
  if ! ask_yes_no "是否安装 ComfyUI requirements？可能下载数 GB PyTorch/CUDA 依赖" y; then
    warn "已跳过 ComfyUI requirements；后续可手动执行："
    warn "  $COMFYUI_PYTHON -m pip install -r $COMFYUI_ROOT/requirements.txt"
    return 0
  fi

  info "安装 ComfyUI requirements（$COMFYUI_ROOT/requirements.txt）"
  "$COMFYUI_PYTHON" -m pip install --upgrade pip >/dev/null 2>&1 || warn "pip 升级失败，继续安装 requirements"
  PIP_DISABLE_PIP_VERSION_CHECK=1 "$COMFYUI_PYTHON" -m pip install -r "$COMFYUI_ROOT/requirements.txt" || {
    error "ComfyUI requirements 安装失败"
    return 1
  }
  ok "ComfyUI 依赖安装完成"
  return 0
}
