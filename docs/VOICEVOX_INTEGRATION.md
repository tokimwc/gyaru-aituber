# VOICEVOX連携実装ドキュメント

## 概要

ギャルAITuber「りんね」にVOICEVOX音声合成機能を統合し、テキスト応答を自動的に音声で読み上げる機能を実装しました。

## システム構成

```
┌─────────────────────────────────────────────────────────────┐
│                        Windows環境                           │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  ブラウザ        │  │  VOICEVOX        │                │
│  │  (Chrome/Edge)   │  │  (v0.14.6)       │                │
│  │                  │  │                  │                │
│  │  http://<WSL2_IP>│  │  http://127.0.0. │                │
│  │  :8080           │  │  1:50021         │                │
│  └────────┬─────────┘  └────────┬─────────┘                │
│           │ WebSocket           │ HTTP API                  │
│           │                     │                           │
└───────────┼─────────────────────┼───────────────────────────┘
            │                     │
            │                     │ fetch()
            │                     │
┌───────────┼─────────────────────┼───────────────────────────┐
│           │    WSL2環境         │                           │
│  ┌────────▼─────────┐           │                           │
│  │  FastAPI Server  │           │                           │
│  │  (server_gyaru.py)│          │                           │
│  │                  │           │                           │
│  │  - WebSocket     │           │                           │
│  │  - RepE Engine   │           │                           │
│  │  - LLM (Qwen2.5) │           │                           │
│  └──────────────────┘           │                           │
│                                 │                           │
└─────────────────────────────────┴───────────────────────────┘
```

## 実装の流れ

### 1. 初期試行：サーバー側プロキシ方式（失敗）

**アプローチ**:
- FastAPIサーバーにVOICEVOX APIへのプロキシエンドポイントを追加
- WSL2からWindowsホスト（`<WINDOWS_HOST_IP>:50021`）にhttpxでリクエスト

**注意**: `<WINDOWS_HOST_IP>`はWSL2環境におけるWindowsホストのIPアドレスです。通常は`172.16.x.1`〜`172.31.x.1`の範囲になります。

**実装コード**:
```python
import httpx

# WSL2環境でのWindowsホストアドレス（環境により異なる）
VOICEVOX_BASE_URL = "http://<WINDOWS_HOST_IP>:50021"

@app.post("/voicevox/audio_query")
async def voicevox_audio_query(text: str, speaker: int = 8):
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{VOICEVOX_BASE_URL}/audio_query",
            params={"text": text, "speaker": speaker}
        )
        return response.json()

@app.post("/voicevox/synthesis")
async def voicevox_synthesis(request: Request, speaker: int = 8):
    query_json = await request.json()
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{VOICEVOX_BASE_URL}/synthesis",
            params={"speaker": speaker},
            json=query_json
        )
        return Response(content=response.content, media_type="audio/wav")
```

**失敗理由**:
```bash
# WSL2からWindowsホストへのアクセステスト
$ curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "http://<WINDOWS_HOST_IP>:50021/version"
000接続失敗
```

- Windowsファイアウォールが WSL2 → Windows方向の通信をブロック
- タイムアウト（60秒）で失敗
- サーバーログ: `VOICEVOX audio_query タイムアウト:`

### 2. 最終実装：ブラウザから直接アクセス方式（成功）

**アプローチ**:
- ブラウザはWindows上で動作するため、`localhost:50021` に直接アクセス可能
- CORS問題を解決するため、VOICEVOX設定を変更

**実装コード（JavaScript）**:
```javascript
const VOICEVOX_API_URL = "http://127.0.0.1:50021";
const SPEAKER_ID = 8; // 春日部つむぎノーマル

async function playVoicevox(text) {
    if (!text) return;
    console.log("VOICEVOX再生開始:", text);

    try {
        // 1. 音声合成用クエリの作成
        const queryResponse = await fetch(
            `${VOICEVOX_API_URL}/audio_query?text=${encodeURIComponent(text)}&speaker=${SPEAKER_ID}`,
            { method: "POST" }
        );
        if (!queryResponse.ok) {
            throw new Error(`Audio Query Failed: ${queryResponse.status}`);
        }
        const queryJson = await queryResponse.json();
        console.log("VOICEVOX クエリ作成成功");

        // 2. 音声データの生成
        const synthesisResponse = await fetch(
            `${VOICEVOX_API_URL}/synthesis?speaker=${SPEAKER_ID}`,
            {
                method: "POST",
                headers: { 
                    "Content-Type": "application/json",
                    "Accept": "audio/wav"
                },
                body: JSON.stringify(queryJson)
            }
        );
        if (!synthesisResponse.ok) {
            throw new Error(`Synthesis Failed: ${synthesisResponse.status}`);
        }
        const audioBlob = await synthesisResponse.blob();
        console.log("VOICEVOX 音声合成成功, サイズ:", audioBlob.size, "bytes");

        // 3. ブラウザで再生
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.volume = 1.0;
        
        audio.onended = () => {
            console.log("VOICEVOX再生完了");
            URL.revokeObjectURL(audioUrl);
        };
        
        audio.onerror = (e) => {
            console.error("音声再生エラー:", e);
            URL.revokeObjectURL(audioUrl);
        };
        
        await audio.play();
        console.log("VOICEVOX音声再生開始");

    } catch (error) {
        console.error("VOICEVOXエラー:", error);
    }
}

// WebSocketメッセージ受信時に呼び出し
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "done") {
        playVoicevox(data.full_response);
    }
};
```

## CORS問題の解決

### 問題

```
Access to fetch at 'http://127.0.0.1:50021/audio_query' from origin 'http://<WSL2_IP>:8080'
has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

### 原因

VOICEVOX 0.14.6のデフォルト設定:
- **CORS Policy Mode**: `localapps`
- `localapps`: `app://` スキーム（Electronアプリ）からのみアクセス許可
- ブラウザ（`http://` スキーム）からのアクセスはブロック

### 解決手順

1. **VOICEVOX設定画面を開く**
   - VOICEVOXアプリ → 設定 → オプション

2. **CORS Policy Modeを変更**
   - `localapps` → `all` に変更
   - 「保存」ボタンをクリック

3. **VOICEVOXを再起動**（重要！）
   - 設定変更は再起動後に反映される

4. **設定確認**
   ```bash
   # ブラウザまたはcurlで確認
   curl http://127.0.0.1:50021/setting
   
   # 出力例
   {
     "cors_policy_mode": "all",
     "allow_origin": null
   }
   ```

### CORS Policy Modeの種類

| Mode | 説明 | 用途 |
|------|------|------|
| `localapps` | `app://` スキームのみ許可 | VOICEVOX公式エディタ専用 |
| `all` | すべてのオリジンを許可 | ブラウザ、外部アプリからの利用 |

参考: [VOICEVOX Engine Issue #392](https://github.com/VOICEVOX/voicevox_engine/issues/392)

## ブラウザ拡張機能の問題

### ERR_BLOCKED_BY_CLIENT エラー

**原因**:
- 広告ブロッカー（uBlock Origin, AdBlock Plus等）が `localhost:50021` へのリクエストをブロック
- 一部の拡張機能は `localhost` や `127.0.0.1` へのリクエストも対象にする

**解決策**:

1. **拡張機能を一時的に無効化**
   - ブラウザの拡張機能管理画面で広告ブロッカーを無効化
   - テスト後に再度有効化

2. **ホワイトリストに追加**
   - uBlock Originの場合:
     - 拡張機能アイコン → 設定 → ホワイトリスト
     - `127.0.0.1` または `localhost` を追加

3. **開発者モードで確認**
   - F12 → Console でエラーメッセージを確認
   - Network タブでリクエストの状態を確認

## 話者（Speaker）の選択

VOICEVOX 0.14.6で利用可能な話者:

| Speaker ID | 名前 | スタイル | 特徴 |
|-----------|------|---------|------|
| 0 | 四国めたん | ノーマル | 標準的な女性声 |
| 2 | 四国めたん | あまあま | 甘い声 |
| 3 | ずんだもん | ノーマル | 中性的な声 |
| 8 | 春日部つむぎ | ノーマル | 元気な女性声（ギャル向き） |
| 10 | 雨晴はう | ノーマル | 落ち着いた女性声 |

**推奨**: Speaker ID `8`（春日部つむぎノーマル）
- 元気でハキハキした声質
- ギャルキャラに適している
- 明瞭な発音

話者一覧の取得:
```bash
curl http://127.0.0.1:50021/speakers
```

## パフォーマンス

### 音声合成時間

- **クエリ作成**: 約1秒
- **音声合成**: 約60秒（文章の長さに依存）
- **合計**: 約61秒

### 最適化案

1. **文章の分割**
   - 長文を句点で分割して並列処理
   - 各セグメントを順次再生

2. **キャッシュ機能**
   - 同じテキストの音声をキャッシュ
   - IndexedDBやLocalStorageに保存

3. **ストリーミング再生**
   - 音声合成完了前に再生開始
   - WebSocket経由でチャンク送信

## トラブルシューティング

### 音声が出ない場合

1. **VOICEVOXが起動しているか確認**
   ```bash
   curl http://127.0.0.1:50021/version
   ```
   正常な場合: `{"version":"0.14.6"}`

2. **CORS設定を確認**
   ```bash
   curl http://127.0.0.1:50021/setting
   ```
   `cors_policy_mode` が `"all"` であることを確認

3. **ブラウザのコンソールを確認**
   - F12 → Console
   - エラーメッセージを確認

4. **ブラウザ拡張機能を無効化**
   - 広告ブロッカーを一時的にオフ

### タイムアウトエラー

**原因**: 音声合成に時間がかかりすぎる

**対策**:
- テキストを短くする（100文字以内推奨）
- VOICEVOXのGPU設定を確認
- 他のアプリケーションを終了してリソースを確保

## 動作確認

### テストフロー

```
1. ユーザー入力: "こんにちは"
   ↓
2. WebSocket送信
   ↓
3. LLM生成: "おっ！りんねだ！あーし、最近のAIで盛り上がってるか？..."
   ↓
4. ブラウザに表示（ストリーミング）
   ↓
5. 生成完了後、VOICEVOX APIを呼び出し
   ↓
6. 音声合成（約60秒）
   ↓
7. ブラウザで音声再生
```

### 確認項目

- [x] WebSocket接続確立
- [x] テキスト生成とストリーミング表示
- [x] ギャル度メーター更新
- [x] VOICEVOX APIへのリクエスト成功
- [x] 音声データの取得
- [x] ブラウザでの音声再生
- [x] メモリ解放（URL.revokeObjectURL）

## まとめ

### 成功した構成

- **バックエンド**: WSL2 (FastAPI + PyTorch + RepE)
- **フロントエンド**: Windows Browser (WebSocket + VOICEVOX API)
- **音声合成**: Windows (VOICEVOX 0.14.6)

### 重要なポイント

1. **CORS Policy Modeを `all` に設定**
2. **VOICEVOXを再起動して設定を反映**
3. **ブラウザ拡張機能を無効化またはホワイトリスト追加**
4. **ブラウザから直接 `localhost:50021` にアクセス**（プロキシ不要）

### 今後の改善

- 音声合成の高速化（文章分割、並列処理）
- キャッシュ機能の実装
- 話者選択のUI実装
- 音量・速度調整機能
