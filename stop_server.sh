#!/bin/bash
# FastAPIサーバー停止スクリプト

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  ギャルAITuber「りんね」サーバー停止"
echo "=========================================="
echo ""

# プロセスを検索
PIDS=$(pgrep -f "server_gyaru.py" || true)

if [ -z "$PIDS" ]; then
    echo "✅ サーバーは起動していません"
    exit 0
fi

echo "🔍 検出されたプロセス:"
echo "$PIDS" | while read pid; do
    ps -p "$pid" -o pid,cmd --no-headers | awk '{print "   PID: " $1 " - " substr($0, index($0,$2))}'
done
echo ""

# プロセスを停止
echo "🛑 サーバーを停止しています..."
pkill -f "server_gyaru.py" || true

# 少し待ってから確認
sleep 2

# 停止確認
REMAINING=$(pgrep -f "server_gyaru.py" || true)
if [ -z "$REMAINING" ]; then
    echo "✅ サーバーを停止しました"
else
    echo "⚠️  一部のプロセスが残っています。強制停止します..."
    kill -9 $REMAINING 2>/dev/null || true
    sleep 1
    FINAL_CHECK=$(pgrep -f "server_gyaru.py" || true)
    if [ -z "$FINAL_CHECK" ]; then
        echo "✅ サーバーを強制停止しました"
    else
        echo "❌ プロセスの停止に失敗しました"
        exit 1
    fi
fi

echo ""
