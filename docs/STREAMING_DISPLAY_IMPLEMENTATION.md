# ストリーミング表示実装ドキュメント

**実装日**: 2026-01-11  
**検証日**: 2026-01-11  
**担当**: Claude in Cursor

---

## 概要

AITuber WebUIにおいて、LLMの生成テキストをトークンごとにリアルタイムで表示するストリーミング機能を実装した。当初、バックエンドが一括生成後に1文字ずつ送信する「偽のストリーミング」だったため、真のストリーミング生成に変更した。

---

## 問題の発見

### 初期症状

- WebUIでテキストが全文生成後に一括表示される
- トークンごとの流れるような表示がされない
- TTSの音声が、全文表示後に開始される

### ユーザーからの指摘

> 「テキスト表示はストリーミングではないような気がします。テキストが全て出力されてから、TTSの音声が開始されています。」

---

## 原因分析

### デバッグプロセス

1. **フロントエンドの確認**
   - `appendToken` 関数のコードを確認
   - JavaScriptの実装は正しかった
   - ブラウザのコンソールログに `type === 'token'` のログが全く出力されない

2. **バックエンドの確認**
   - サーバーログを確認
   - WebSocket接続は確立されている
   - しかし、トークン送信のログも出力されていない

3. **生成関数の確認**
   - `generate_response` 関数を調査
   - **真の原因発見**: `model.generate()` で全トークンを一括生成していた

### 「偽のストリーミング」の構造

```python
# 旧実装（偽のストリーミング）
def generate_response(messages, max_tokens=128):
    # 一括生成
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, ...)
    
    # デコード
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return generated_text

# WebSocketハンドラ
full_response = await loop.run_in_executor(None, generate_response, messages, 128)

# 生成完了後、1文字ずつ送信（偽のストリーミング）
for i, char in enumerate(full_response):
    await websocket.send_json({"type": "token", "content": char})
    await asyncio.sleep(0.01)  # 遅延を追加
```

**問題点**:
- `model.generate()` は全トークンを生成してから返す
- 生成が完了するまで、WebSocketへの送信が開始されない
- 1文字ずつの送信は、生成完了後のシミュレーションに過ぎない

---

## 解決策

### 1. ストリーミング生成関数の実装

`model.generate()` の代わりに、`model()` を直接呼び、1トークンずつ生成してyieldする関数を実装。

```python
def generate_response_streaming(messages: list, max_tokens: int = 128):
    """
    モデルで応答を生成（ストリーミング版 - ジェネレーター）
    
    Yields:
        str: 生成されたトークン（デコード済み文字列）
    """
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        past_key_values = None
        generated_tokens = []
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        
        for _ in range(max_tokens):
            # 1トークンずつ生成
            if past_key_values is None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True
                )
            else:
                outputs = model(
                    input_ids=next_token_id,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            past_key_values = outputs.past_key_values
            
            # 次のトークンをサンプリング
            logits = outputs.logits[:, -1, :]
            logits = logits / 0.7  # temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # EOSトークンチェック
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token_id.item())
            
            # トークンをデコードして yield
            token_text = tokenizer.decode([next_token_id.item()], skip_special_tokens=True)
            if token_text:
                yield token_text
            
            # attention_maskを更新
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)
```

**技術的ポイント**:
- `past_key_values`: 前のトークンの計算結果を再利用して高速化
- `temperature=0.7`: 多様性を確保
- `top-p sampling (p=0.9)`: 累積確率でトークンをフィルタリング
- `tokenizer.eos_token_id`: 生成終了を判定

### 2. 非同期Queue方式の採用

別スレッドでストリーミング生成を実行し、`asyncio.Queue` でトークンをリアルタイムにWebSocketへ送信。

```python
# ストリーミング生成用のQueue
token_queue = asyncio.Queue()
full_response = ""

def generate_in_thread():
    """別スレッドでストリーミング生成を実行し、Queueに送信"""
    try:
        logger.info("🔄 generate_in_thread 開始")
        for token in generate_response_streaming(messages, 128):
            logger.info(f"🎯 トークン生成: {repr(token)}")
            asyncio.run_coroutine_threadsafe(
                token_queue.put(token),
                loop
            )
    except Exception as e:
        logger.error(f"❌ 生成エラー: {e}", exc_info=True)
    finally:
        logger.info("✅ 生成完了、終了シグナル送信")
        asyncio.run_coroutine_threadsafe(
            token_queue.put(None),  # 終了シグナル
            loop
        )

loop = asyncio.get_event_loop()
executor_task = loop.run_in_executor(None, generate_in_thread)

logger.info("⏳ トークン受信待機中")

# トークンをリアルタイムで送信
while True:
    token = await token_queue.get()
    if token is None:  # 終了シグナル
        logger.info("🛑 終了シグナル受信")
        break
    
    logger.info(f"📤 トークン送信: {repr(token)}")
    full_response += token
    await websocket.send_json({
        "type": "token",
        "content": token
    })

await executor_task  # スレッドの完了を待つ

logger.info(f"🎉 生成完了: {len(full_response)}文字")

# 生成完了を通知
await websocket.send_json({
    "type": "done",
    "full_response": full_response
})
```

**技術的ポイント**:
- `asyncio.run_coroutine_threadsafe()`: 別スレッドからasyncioのコルーチンを安全に呼び出し
- `asyncio.Queue`: スレッド間のリアルタイム通信
- 終了シグナル (`None`): 生成完了を通知

### 3. デバッグログの追加

バックエンドとフロントエンドの両方にデバッグログを追加して、問題箇所を特定できるようにした。

**バックエンド**:
```python
logger.info("🔄 generate_in_thread 開始")
logger.info(f"🎯 トークン生成: {repr(token)}")
logger.info(f"📤 トークン送信: {repr(token)}")
logger.info("✅ 生成完了、終了シグナル送信")
logger.info("🛑 終了シグナル受信")
logger.info(f"🎉 生成完了: {len(full_response)}文字")
```

**フロントエンド**:
```javascript
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`[WS受信] type=${data.type}, content=${data.content ? data.content.substring(0, 20) : 'N/A'}`);
    
    if (data.type === 'token') {
        console.log(`[TOKEN受信] ${JSON.stringify(data.content)}`);
        appendToken(data.content);
        // ...
    }
};
```

---

## 検証結果

### バックエンドログ

トークンが50ms間隔で生成・送信されている:

```
2026-01-11 12:11:22,241 - INFO - 🎯 トークン生成: 'な'
2026-01-11 12:11:22,242 - INFO - 📤 トークン送信: 'な'
2026-01-11 12:11:22,291 - INFO - 🎯 トークン生成: 'のは'
2026-01-11 12:11:22,292 - INFO - 📤 トークン送信: 'のは'
2026-01-11 12:11:22,350 - INFO - 🎯 トークン生成: '、'
2026-01-11 12:11:22,350 - INFO - 📤 トークン送信: '、'
2026-01-11 12:11:22,400 - INFO - 🎯 トークン生成: '少し'
2026-01-11 12:11:22,400 - INFO - 📤 トークン送信: '少し'
```

### フロントエンドログ

トークンをリアルタイムで受信:

```
[LOG] [WS受信] type=token, content=お
[LOG] [TOKEN受信] "お"
[LOG] [WS受信] type=token, content=っ
[LOG] [TOKEN受信] "っ"
[LOG] [WS受信] type=token, content=しゃ
[LOG] [TOKEN受信] "しゃ"
[LOG] [WS受信] type=gauge, content=N/A
[LOG] [WS受信] type=token, content=る
[LOG] [TOKEN受信] "る"
[LOG] [WS受信] type=token, content=通り
[LOG] [TOKEN受信] "通り"
```

### UI表示

- ✅ 文字が1つずつ流れるように表示される
- ✅ ChatGPTライクな応答性
- ✅ VOICEVOX連携: 句読点検出で音声合成が並行動作

### パフォーマンス

- **トークン生成速度**: 約50-60ms/token（14Bモデル、RTX 5090）
- **VRAM使用量**: 19.3GB（4bit量子化）
- **体感的な応答性**: 優れている（真のストリーミング）

---

## 学び

### 1. `model.generate()` は一括生成のみ

HuggingFace Transformersの `model.generate()` は、全トークンを生成してから返すため、リアルタイムストリーミングには不向き。

### 2. 手動トークン生成が必要

`model()` を直接呼び、1トークンずつサンプリングする必要がある。past_key_valuesを使えば、計算効率も高い。

### 3. asyncio.Queueが効果的

別スレッドでの生成結果をリアルタイムにWebSocketへ送信するには、`asyncio.Queue` が最適。

### 4. デバッグログの重要性

バックエンド/フロントエンドの両方でログを出力することで、問題箇所を迅速に特定できる。

### 5. 「偽のストリーミング」に注意

全文生成後の1文字ずつ送信は、真のストリーミングではない。ユーザー体験が大きく異なる。

---

## 今後の課題

### 1. パフォーマンス最適化

- KV-Cacheの最適化
- バッチ推論の検討
- GPU利用率の向上

### 2. エラーハンドリング

- 生成中のエラーをフロントエンドに通知
- リトライ機能の実装
- タイムアウト処理

### 3. 機能拡張

- トークン単位ではなく、単語単位での送信オプション
- 生成速度の動的調整
- ユーザーによる生成停止機能

---

## 参考資料

- [HuggingFace Transformers - Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [asyncio - Asynchronous I/O](https://docs.python.org/3/library/asyncio.html)
- [FastAPI - WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)

---

## 日本語文字化け問題の修正 (2026-01-12)

### 問題発見

ストリーミング表示実装後、日本語が文字化けする現象が発生：

```
かし��まりました。すぐにお茶を淹れて参りますので、少し��待ちください。
```

### 原因

Qwen2.5のトークナイザーは**バイトレベルBPE**で動作するため、日本語の1文字が複数のトークンに分割されることがある。1トークンずつデコードすると、不完全なUTF-8バイト列が `\ufffd`（置換文字）として表示される。

### 解決策

**累積デコード + 末尾不完全文字待機**の手法を採用：

```python
# 累積トークンをデコード
current_text = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

# 不完全なUTF-8文字（置換文字 U+FFFD）が末尾にある場合は待機
if current_text.endswith('\ufffd'):
    # 末尾の不完全な文字を除いた部分だけを出力対象にする
    safe_text = current_text.rstrip('\ufffd')
else:
    safe_text = current_text

if len(safe_text) > len(previous_text):
    new_text = safe_text[len(previous_text):]
    yield new_text
    previous_text = safe_text
```

### 結果

- ✅ 文字化けなし
- ✅ 文字欠落なし
- ✅ 正しく「かしこまりました」が表示される

詳細は [`JAPANESE_TOKENIZATION_FIX.md`](JAPANESE_TOKENIZATION_FIX.md) を参照。

---

## まとめ

真のストリーミング生成を実装することで、ユーザー体験が大幅に向上した。`model.generate()` の一括生成から、手動トークン生成 + asyncio.Queue方式への変更が成功の鍵だった。デバッグログの追加により、問題の特定も迅速に行えた。

さらに、日本語文字化け問題を累積デコード + 末尾不完全文字待機の手法で解決し、完璧なストリーミング表示を実現した。今後は、パフォーマンス最適化とエラーハンドリングの強化が課題となる。
