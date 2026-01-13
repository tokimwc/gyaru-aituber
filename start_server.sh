#!/bin/bash
# FastAPIサーバー起動スクリプト
# VOICEVOX連携機能を含むギャルAITuber「りんね」サーバー

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  ギャルAITuber「りんね」サーバー起動"
echo "=========================================="
echo ""

# 既に起動しているかチェック
if pgrep -f "server_gyaru.py" > /dev/null; then
    echo "⚠️  既にサーバーが起動しています"
    echo "   停止するには: ./stop_server.sh"
    exit 1
fi

echo "📦 依存パッケージを確認中..."
echo ""

# サーバー起動
echo "🚀 サーバーを起動しています..."
echo "   URL: http://0.0.0.0:8080"
echo "   Windows側から: http://<WSL2_IP>:8080"
echo "   （WSL2 IPアドレス確認: hostname -I）"
echo ""
echo "   ⚠️  モデルロードには1-2分かかります"
echo "   ⚠️  停止するには: Ctrl+C または ./stop_server.sh"
echo ""
echo "=========================================="
echo ""

# Strength引数が指定されている場合は使用、なければデフォルト10.0
STRENGTH="${1:-10.0}"

echo "   Strength: ${STRENGTH}"
echo ""

uv run --with fastapi --with uvicorn --with websockets --with httpx --with transformers --with accelerate --with bitsandbytes --with pyyaml python src/server_gyaru.py --strength "${STRENGTH}"
