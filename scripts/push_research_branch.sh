#!/usr/bin/env bash
# 一键推送 research/action-response-v1 分支到 GitHub
#
# 用法:
#   ./scripts/push_research_branch.sh                # 交互输入 Token 并推送
#   ./scripts/push_research_branch.sh --dry-run      # 只验证认证，不实际推送
#   GITHUB_TOKEN=xxx ./scripts/push_research_branch.sh   # 从环境变量读 Token
#   ./scripts/push_research_branch.sh --force-with-lease # 远端分支已存在且分叉时
#   ./scripts/push_research_branch.sh --store        # 顺便保存凭据(明文，慎用)
#
# 说明:
#   - 自动检测 GitHub 直连，不通则走 ghfast.top 代理（本机实测直连不可达）
#   - Token 仅通过临时 GIT_ASKPASS 传给 git，不写入磁盘、不出现在进程参数里
#   - Token 需要的权限: classic PAT 勾选 repo（或 fine-grained 的 Contents: Read and write）
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="research/action-response-v1"
OWNER="bright-moon-67"
REPO="H-JEPA"
DIRECT="https://github.com/${OWNER}/${REPO}.git"
PROXY="https://ghfast.top/https://github.com/${OWNER}/${REPO}.git"

FORCE=""
STORE=0
DRY=0
for arg in "$@"; do
  case "$arg" in
    --force-with-lease) FORCE="--force-with-lease" ;;
    --store) STORE=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,16p' "$0" | grep -v '^set -euo pipefail$'; exit 0 ;;
    *) echo "未知参数: $arg（用 --help 查看用法）"; exit 2 ;;
  esac
done

cd "$REPO_DIR"
git rev-parse --verify "refs/heads/$BRANCH" >/dev/null 2>&1 || {
  echo "✗ 本地不存在分支 $BRANCH"; exit 1; }

# ---- 选择推送通道 ----
REMOTE="$DIRECT"
if ! GIT_TERMINAL_PROMPT=0 timeout 15 git ls-remote --heads "$REMOTE" >/dev/null 2>&1; then
  echo "→ GitHub 直连不可达，改用 ghfast.top 代理通道"
  REMOTE="$PROXY"
fi
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url --push origin "$REMOTE"
else
  git remote add origin "$REMOTE"
fi

echo
echo "→ 本次将推送以下提交:"
git --no-pager log --oneline -4 "$BRANCH"
echo

# ---- 获取凭据（不落盘）----
USER_NAME="${GITHUB_USER:-$OWNER}"
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  TOKEN="$GITHUB_TOKEN"
  echo "→ 使用环境变量 GITHUB_TOKEN 中的凭据"
else
  read -r -p "GitHub 用户名 [$USER_NAME]: " INPUT_NAME
  USER_NAME="${INPUT_NAME:-$USER_NAME}"
  read -r -s -p "粘贴 GitHub Token（输入不回显）: " TOKEN
  echo
fi
[[ -n "$TOKEN" ]] || { echo "✗ 未提供 Token"; exit 1; }

ASKPASS="$(mktemp)"
chmod 700 "$ASKPASS"
cat > "$ASKPASS" <<'ASKPASS_EOF'
#!/usr/bin/env bash
case "$1" in
  *sername*) printf '%s\n' "${HJ_USER:?}" ;;
  *)         printf '%s\n' "${HJ_TOKEN:?}" ;;
esac
ASKPASS_EOF

cleanup() {
  rm -f "$ASKPASS"
  unset HJ_USER HJ_TOKEN GIT_ASKPASS || true
}
trap cleanup EXIT

export GIT_ASKPASS="$ASKPASS" HJ_USER="$USER_NAME" HJ_TOKEN="$TOKEN"
export GIT_TERMINAL_PROMPT=0

PUSH_ARGS=(-u origin "$BRANCH")
[[ -n "$FORCE" ]] && PUSH_ARGS=("$FORCE" "${PUSH_ARGS[@]}")
[[ $DRY -eq 1 ]] && PUSH_ARGS=(--dry-run "${PUSH_ARGS[@]}")

if ! git push "${PUSH_ARGS[@]}"; then
  cat <<'MSG'

✗ 推送失败，常见原因：
  1) Token 无写权限 → 重新生成 classic PAT 并勾选 repo；
  2) 远端已有同名分叉分支 → 加 --force-with-lease 重试；
  3) 代理不稳定 → 稍后重试，或改用 GITHUB_TOKEN=xxx 方式运行。
MSG
  exit 1
fi

if [[ $STORE -eq 1 && $DRY -eq 0 ]]; then
  git config --global credential.helper store
  printf 'https://%s:%s@github.com\n' "$USER_NAME" "$TOKEN" >> "$HOME/.git-credentials"
  chmod 600 "$HOME/.git-credentials"
  echo "→ 凭据已保存到 ~/.git-credentials（明文存储，注意安全）"
fi

if [[ $DRY -eq 1 ]]; then
  echo "✓ 认证通过（dry-run），未实际推送"
  exit 0
fi

echo
echo "✓ 推送完成"
echo "  分支页面: https://github.com/${OWNER}/${REPO}/tree/${BRANCH}"
echo "  发起 PR:  https://github.com/${OWNER}/${REPO}/compare/main...${BRANCH}?expand=1"
