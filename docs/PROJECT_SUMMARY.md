# ギャルAITuber「りんね」プロジェクト - 完成報告

## プロジェクト概要

Representation Engineering（RepE）を用いて、大規模言語モデル（Qwen2.5-32B）の性格を「標準的なAI」から「あーし系技術強者ギャル」へ変貌させ、リアルタイムで「思考の可視化」を行うAITuberシステムを構築しました。

## 最終成果物

### 1. システム構成

```
┌─────────────────────────────────────────────────────────────┐
│                        Windows環境                           │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  ブラウザ        │  │  VOICEVOX        │                │
│  │  (OBS対応)       │  │  (v0.14.6)       │                │
│  │                  │  │                  │                │
│  │  Cyberpunk UI    │  │  音声合成        │                │
│  │  ギャル度メーター│  │  Speaker ID: 8   │                │
│  └────────┬─────────┘  └────────┬─────────┘                │
│           │ WebSocket           │ HTTP API                  │
│           │ (リアルタイム)      │ (音声生成)                │
└───────────┼─────────────────────┼───────────────────────────┘
            │                     │
            │                     │
┌───────────┼─────────────────────┼───────────────────────────┐
│           │    WSL2環境         │                           │
│  ┌────────▼─────────┐           │                           │
│  │  FastAPI Server  │           │                           │
│  │                  │           │                           │
│  │  ┌────────────┐  │           │                           │
│  │  │ RepE Engine│  │           │                           │
│  │  │ (PyTorch)  │  │           │                           │
│  │  │            │  │           │                           │
│  │  │ - Pre-Hook │  │           │                           │
│  │  │ - Vector   │  │           │                           │
│  │  │   Injection│  │           │                           │
│  │  │ - Cosine   │  │           │                           │
│  │  │   Similarity│ │           │                           │
│  │  └────────────┘  │           │                           │
│  │                  │           │                           │
│  │  ┌────────────┐  │           │                           │
│  │  │ LLM        │  │           │                           │
│  │  │ Qwen2.5-32B│  │           │                           │
│  │  │ (bnb-4bit) │  │           │                           │
│  │  └────────────┘  │           │                           │
│  └──────────────────┘           │                           │
│                                 │                           │
└─────────────────────────────────┴───────────────────────────┘
```

### 2. 主要機能

#### 2.1 Representation Engineering（RepE）
- **データセット**: 117ペアの「標準語 ⇔ ギャル語」対話データ
- **ベクトル抽出**: PyTorch Hookを用いた手動実装（repengライブラリの互換性問題を回避）
- **ベクトル注入**: `register_forward_pre_hook` で31レイヤーに介入
- **強度**: Strength 15.0（Pre-Hookでの正規化減衰を考慮）

#### 2.2 リアルタイム可視化
- **ギャル度メーター**: コサイン類似度をネオンバーで表示
- **更新頻度**: トークン生成ごと（約50ms間隔）
- **値の範囲**: 0.3〜0.5（標準的なギャル会話）

#### 2.3 音声合成（VOICEVOX）
- **話者**: 春日部つむぎノーマル（Speaker ID: 8）
- **処理時間**: 約60秒/100文字
- **フロー**: テキスト生成完了 → 音声合成 → 自動再生

#### 2.4 WebUI
- **デザイン**: Cyberpunk風ネオンUI
- **レスポンシブ**: OBS Browser Source対応
- **通信**: WebSocketによるリアルタイムストリーミング

## 技術的成果

### 1. RepE実装の革新

**従来の問題**:
- repengライブラリがQwen2.5に非対応
- `AttributeError: 'ControlModule' object has no attribute 'attention_type'`

**解決策**:
- PyTorch Hookを直接使用した手動実装
- scikit-learnのPCAで主成分分析
- ライブラリ依存なしで完全制御

**コード例**:
```python
def apply_gyaru_pre_hook(vector: torch.Tensor, strength: float):
    def hook(module, args):
        hidden_states = args[0]
        intervention = vector * strength
        intervention = intervention.to(hidden_states.dtype)
        new_hidden = hidden_states + intervention
        return (new_hidden,) + args[1:]
    return hook
```

### 2. Pre-Hook vs Forward Hook

| 特性 | Forward Hook | Pre-Hook |
|------|-------------|----------|
| 介入タイミング | レイヤー出力後 | レイヤー入力前 |
| 型エラー | 発生しやすい | 発生しにくい |
| 最適Strength | 1.5〜2.0 | 10.0〜20.0 |
| 効果 | 直接的 | 正規化で減衰 |
| 安定性 | 低い | 高い |

**結論**: Pre-Hookの方が安全で効果的

### 3. ハイブリッド制御

**System Prompt + Vector = 最強**

```python
system_prompt = "あなたは快活なAITuber「りんね」です。一人称は必ず「あーし」を使ってください。"
# + RepEベクトル（Strength 15.0）
```

**効果**:
- 「あーし」使用率: 100%
- ギャル口調: 安定
- キャラクター一貫性: 高い

### 4. WSL2/Windows統合

**課題**:
- WSL2 → Windows方向の通信がファイアウォールでブロック
- VOICEVOX APIへのアクセス不可

**解決策**:
- ブラウザから直接 `localhost:50021` にアクセス
- CORS Policy Modeを `all` に設定
- プロキシ不要のシンプルな構成

## パフォーマンス

### 1. モデル推論

| 項目 | 値 |
|------|-----|
| モデル | Qwen2.5-32B-Instruct-bnb-4bit |
| VRAM使用量 | 約20GB（推論時） |
| 生成速度 | 約10トークン/秒 |
| レスポンス時間 | 約5秒/100トークン |

### 2. 音声合成

| 項目 | 値 |
|------|-----|
| クエリ作成 | 約1秒 |
| 音声合成 | 約60秒/100文字 |
| 合計 | 約61秒 |

### 3. WebSocket通信

| 項目 | 値 |
|------|-----|
| ゲージ更新頻度 | 50ms |
| トークン送信頻度 | リアルタイム |
| 遅延 | <100ms |

## 検証結果

### 1. RepE効果の定量評価

**ギャル度メーター（コサイン類似度）**:
- 基本挨拶: 0.37
- 技術質問: 0.48（高め）
- カジュアル雑談: 0.42

**解釈**:
- 0.3〜0.5が「標準的なギャル会話」の範囲
- 技術的な話題で値が上昇 = ベクトルが話題に反応

### 2. 生成品質

**「あーし」使用率**: 100%（全レスポンスで確認）

**テストケース例**:
```
入力: "こんにちは"
出力: "おっ！りんねだ！あーし、最近のAIで盛り上がってるか？
      技術の進んだ世界に住んでるなあ、って思ったら、
      あーわけないけど、AIが日常の一部になってるよ。"
```

**特徴**:
- ✅ 「あーし」使用
- ✅ ギャル口調（「〜じゃん」「〜だよ」）
- ✅ テンション高め
- ✅ 自己主張強め

### 3. Geminiによる評価

> 「サイバーパンク × ギャル」の世界観が完璧に表現されている。
> 技術的な話題で0.48に上昇するのは、「難しい話をするときほど強がる」
> RepEの特性を可視化している。

## 開発の軌跡

### Phase 1: データセット生成（1日目）
- vLLM環境構築
- プロンプトエンジニアリング
- 117ペアのデータ生成
- 「あーし」含有率: 0% → 92%

### Phase 2: ベクトル抽出（2日目）
- repeng互換性問題に遭遇
- 手動実装への切り替え
- 31レイヤー分のベクトル抽出成功

### Phase 3: AITuberバックエンド（3日目）
- Forward Hook → Pre-Hookへの移行
- Strength調整（1.5 → 15.0）
- 可視化機能の実装

### Phase 4: WebUI統合（4日目）
- FastAPI + WebSocket実装
- Cyberpunk風UIデザイン
- WSL2/Windows統合テスト

### Phase 5: VOICEVOX連携（5日目）
- CORS問題の解決
- ブラウザ拡張機能対応
- 音声出力成功

## 学んだこと

### 1. ライブラリ互換性問題
- 最新モデルでは既存ライブラリが動かないことがある
- 手動実装の方が確実で柔軟
- RepEの本質を理解していれば実装可能

### 2. Pre-Hookの優位性
- Forward Hookより安全で効果的
- 型エラーが発生しにくい
- Strengthは10倍必要だが安定

### 3. ハイブリッド制御
- System Prompt（骨組み）+ Vector（魂）
- 両方を組み合わせることで最高の効果

### 4. WSL2/Windows統合
- ファイアウォール問題に注意
- ブラウザ経由が最もシンプル
- CORS設定の理解が重要

### 5. VOICEVOX連携
- CORS Policy Modeの設定が必須
- 再起動しないと反映されない
- ブラウザ拡張機能の影響に注意

## 今後の展望

### 短期（1ヶ月以内）
- [ ] 音声合成の高速化（文章分割、並列処理）
- [ ] キャッシュ機能の実装
- [ ] Strength調整UI
- [ ] 話者選択機能

### 中期（3ヶ月以内）
- [ ] 複数キャラクターの実装
- [ ] 感情ベクトルの追加
- [ ] リアルタイム学習機能
- [ ] OBS連携の最適化

### 長期（6ヶ月以内）
- [ ] マルチモーダル対応（画像認識）
- [ ] 3Dアバター連携
- [ ] コミュニティ機能
- [ ] 商用利用の検討

## ファイル構成

```
gyaru-aituber/
├── src/
│   ├── generate_dataset.py          # データセット生成
│   ├── extract_gyaru_vector.py      # ベクトル抽出（手動実装）
│   ├── run_gyaru_aituber.py         # コンソール版AITuber
│   └── server_gyaru.py              # FastAPI WebSocketサーバー
├── outputs/
│   ├── processed/
│   │   └── gyaru_dataset_*.json     # 生成データセット
│   └── vectors/
│       └── *_gyaru_vector_manual.pt # 抽出ベクトル
├── docs/
│   ├── IMPLEMENTATION.md            # 実装詳細
│   ├── VECTOR_EXTRACTION.md         # ベクトル抽出
│   ├── AITUBER_BACKEND.md           # バックエンド実装
│   ├── WEBSOCKET_INTEGRATION.md     # WebSocket統合
│   ├── VOICEVOX_INTEGRATION.md      # VOICEVOX連携
│   ├── WEBUI_VERIFICATION_20260109.md  # WebUI統合テスト (2026-01-09)
│   └── PROJECT_SUMMARY.md           # このファイル
├── start_server.sh                  # サーバー起動スクリプト
├── stop_server.sh                   # サーバー停止スクリプト
└── AGENTS.md                        # 開発記録
```

## 使用方法

### 1. サーバー起動

```bash
cd /path/to/gyaru-aituber
./start_server.sh
```

### 2. VOICEVOX起動

1. Windows上でVOICEVOXを起動
2. 設定 → オプション → CORS Policy Mode を `all` に変更
3. VOICEVOXを再起動

### 3. ブラウザでアクセス

```
http://<WSL2_IP>:8080
```

**注意**: `<WSL2_IP>`はWSL2のIPアドレスです。`hostname -I`コマンドで確認できます。

### 4. OBS統合

1. OBS Studio → ソース追加 → ブラウザ
2. URL: `http://<WSL2_IP>:8080`
3. 幅: 1920、高さ: 1080

## まとめ

このプロジェクトは、Representation Engineeringの実用化と、AITuberシステムへの統合を実現しました。

**主な成果**:
- ✅ RepEの手動実装による柔軟性の向上
- ✅ リアルタイム可視化による「思考の見える化」
- ✅ VOICEVOX連携による音声出力
- ✅ WSL2/Windows統合による実用性の確保
- ✅ OBS対応による配信利用可能

**技術的革新**:
- Pre-Hookによる安全なベクトル注入
- ハイブリッド制御（Prompt + Vector）
- WebSocketによるリアルタイム通信
- クロスOS統合（WSL2 ⇔ Windows）

**今後の可能性**:
- 複数キャラクターへの展開
- 感情ベクトルの追加
- マルチモーダル対応
- 商用利用の検討

---

**プロジェクト完了日**: 2026-01-09  
**開発期間**: 5日間  
**開発者**: @ts-klassen (with Claude)  
**ライセンス**: MIT
