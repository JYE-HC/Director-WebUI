#!/usr/bin/env bash
# shellcheck shell=bash
# Director Web project installation and verification steps.

export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/.uv-cache}"
export PATH="$TOOLS_BIN_DIR:$PATH"

install_director_local() {
  local args=(--comfyui-root "$COMFYUI_ROOT" --node-bin-dir "$NODE_BIN_DIR")
  [[ -n "${COMFYUI_PYTHON_OPTION:-}" ]] && args+=(--comfy-python "$COMFYUI_PYTHON_OPTION")
  [[ "$ASSUME_YES" == true ]] && args+=(-y)
  if ((${#REPLACE_NODES[@]})); then
    local node
    for node in "${REPLACE_NODES[@]}"; do args+=(--replace-node "$node"); done
  fi
  [[ "${CONFIRM_COMFYUI_STOPPED:-false}" == true ]] && args+=(--confirm-comfyui-stopped)
  info "调用现有 install.sh 安装 Director、前端依赖与 bundled custom nodes"
  (cd "$SCRIPT_DIR" && ./install.sh install "${args[@]}")
}

install_director_remote_only() {
  info "安装 Director 后端依赖：uv sync --frozen --no-dev"
  (cd "$SCRIPT_DIR" && "$UV_BIN" sync --frozen --no-dev)
  info "安装并构建前端：npm ci && npm run build"
  (cd "$SCRIPT_DIR/frontend" && "$NODE_BIN_DIR/npm" ci --no-audit --no-fund)
  (cd "$SCRIPT_DIR/frontend" && "$NODE_BIN_DIR/npm" run build)
}

step_install_director() {
  export PATH="$NODE_BIN_DIR:$TOOLS_BIN_DIR:$PATH"
  [[ -x "$UV_BIN" ]] || { error "uv 不可用：$UV_BIN"; return 1; }
  [[ -x "$NODE_BIN_DIR/node" && -x "$NODE_BIN_DIR/npm" ]] || {
    error "Node/npm 不可用：$NODE_BIN_DIR"
    return 1
  }

  if [[ "$COMFYUI_MODE" == skip ]]; then
    install_director_remote_only || { error "Director 依赖安装失败"; return 1; }
    ok "Director 前后端安装完成（使用远程 ComfyUI 模式）"
    return 0
  fi

  [[ -f "$SCRIPT_DIR/install.sh" ]] || {
    error "缺少 install.sh；请使用 director-main 发布包"
    return 1
  }
  install_director_local || {
    error "Director/custom nodes 安装失败；修复 ComfyUI 或节点冲突后可用 --resume 重试"
    error "若上方提示节点“已有内容不同”且确认替换：先 ./bootstrap.sh stop-comfyui，再执行"
    error "  ./bootstrap.sh --only install_director --replace-node <节点名> --confirm-comfyui-stopped"
    error "（旧目录自动备份到 ComfyUI 的 .director-backups/，不会被删除）"
    return 1
  }
  ok "Director 与 custom nodes 安装完成"
  return 0
}

step_verify_offline() {
  if [[ "$COMFYUI_MODE" != skip ]]; then
    local verify_args=(--comfyui-root "$COMFYUI_ROOT" --node-bin-dir "${NODE_BIN_DIR:-}")
    [[ -n "${COMFYUI_PYTHON_OPTION:-}" ]] && verify_args+=(--comfy-python "$COMFYUI_PYTHON_OPTION")
    info "执行 install.sh verify（离线）"
    (cd "$SCRIPT_DIR" && ./install.sh verify "${verify_args[@]}") || {
      error "离线验证失败"
      return 1
    }
  else
    [[ -x "$SCRIPT_DIR/.venv/bin/director-web" ]] || { error "后端入口缺失，安装可能未完成"; return 1; }
    [[ -d "$SCRIPT_DIR/frontend/dist" ]] || { error "前端 dist 缺失，构建可能未完成"; return 1; }
    "$SCRIPT_DIR/.venv/bin/python" -c 'import director' || { error "director Python 包导入失败"; return 1; }
    ok "远程模式下 Director 前后端基础验证通过"
  fi

  return 0
}

step_verify_online_optional() {
  [[ -n "$COMFYUI_URL" ]] || return 0
  [[ -f "$SCRIPT_DIR/tools/check_comfy_online.py" ]] || return 0
  [[ -x "$SCRIPT_DIR/.venv/bin/python" ]] || { warn "后端未安装，跳过 ComfyUI 在线检查"; return 0; }
  info "检查 ComfyUI 在线接口：$COMFYUI_URL"
  if "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/tools/check_comfy_online.py" --url "$COMFYUI_URL"; then
    ok "ComfyUI 在线检查通过"
    return 0
  fi
  if [[ "${REQUIRE_COMFY_ONLINE:-false}" == true ]]; then
    error "ComfyUI 在线检查失败"
    return 1
  fi
  warn "ComfyUI 在线检查失败；安装继续，稍后可在系统设置中重新测试连接"
  return 0
}
