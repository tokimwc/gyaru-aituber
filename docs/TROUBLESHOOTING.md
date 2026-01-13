# トラブルシューティングガイド

このドキュメントでは、プロジェクト実行中に発生する可能性のある問題とその解決方法をまとめています。

## 目次

1. [VRAM不足エラー](#vram不足エラー)
2. [vLLMサーバー起動エラー](#vllmサーバー起動エラー)
3. [モデルロードエラー](#モデルロードエラー)
4. [API接続エラー](#api接続エラー)
5. [データ生成エラー](#データ生成エラー)
6. [よくあるエラーと解決策](#よくあるエラーと解決策)

---

## VRAM不足エラー

### 症状

```
ValueError: Free memory on device (30.02/31.84 GiB) on startup is less than desired GPU memory utilization (0.95, 30.25 GiB).
```

または

```
RuntimeError: Engine core initialization failed.
```

### 原因

- モデルが大きすぎる（32Bモデルは32GB VRAMギリギリ）
- KVキャッシュ用のメモリが不足
- 他のプロセスがVRAMを使用している

### 解決策

#### 1. GPUメモリ使用率を下げる

`config/generation_config.yaml` を編集:

```yaml
vllm:
  gpu_memory_utilization: 0.80  # 0.9 → 0.8に下げる
```

#### 2. 最大シーケンス長を減らす

```yaml
vllm:
  max_model_len: 4096  # 8192 → 4096に下げる
```

#### 3. より軽量なモデルに切り替え

```yaml
model:
  path: "models/Qwen2.5-32B-Instruct-bnb-4bit"
  quantization: "bitsandbytes"  # GPTQ → bitsandbytes
```

#### 4. 他のプロセスを終了

```bash
# GPU使用プロセスを確認
nvidia-smi

# 不要なプロセスを終了
pkill -9 -f "python.*vllm"
```

#### 5. モデルサイズを確認

```bash
# モデルディレクトリのサイズを確認
du -sh models/Qwen3-32B-GPTQ-Int8
```

**推奨**: 32GB VRAMの場合は4bit量子化モデル（約16GB）を使用。

---

## vLLMサーバー起動エラー

### 症状1: bitsandbytesモジュールが見つからない

```
ImportError: Please install bitsandbytes>=0.46.1 via `pip install bitsandbytes>=0.46.1` to use bitsandbytes quantizer.
```

**解決策**:

`src/start_vllm_server.py` のコマンドに `--with bitsandbytes` を追加:

```python
cmd = [
    "uv", "run", "--with", "vllm", "--with", "bitsandbytes", "python", "-m", "vllm.entrypoints.openai.api_server",
    # ...
]
```

### 症状2: dtypeエラー

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig
Value error, torch.bfloat16 is not supported for quantization method gptq. Supported dtypes: [torch.float16]
```

**解決策**:

`src/start_vllm_server.py` のコマンドに `--dtype float16` を追加:

```python
cmd = [
    # ...
    "--dtype", "float16",  # GPTQはfloat16のみサポート
]
```

### 症状3: モデルパスが見つからない

```
FileNotFoundError: Model path does not exist: models/...
```

**解決策**:

1. モデルパスを確認:
   ```bash
   ls -la models/
   ```

2. `config/generation_config.yaml` のパスを修正:
   ```yaml
   model:
     path: "models/正しいパス"
   ```

3. WSL2ではWindowsパスは `/mnt/` で始まる:
   - `YOUR_MODEL_PATH` → `models/...`

---

## モデルロードエラー

### 症状: trust_remote_codeエラー

```
ValueError: You are trying to instantiate a generic tokenizer class with the class `QWenTokenizer`.
```

**解決策**:

`config/generation_config.yaml` で `trust_remote_code: true` を設定:

```yaml
vllm:
  trust_remote_code: true
```

`src/start_vllm_server.py` で `--trust-remote-code` フラグを追加:

```python
if vllm_config['trust_remote_code']:
    cmd.append("--trust-remote-code")
```

---

## API接続エラー

### 症状1: 接続拒否

```
openai.APIConnectionError: Connection error.
httpcore.ConnectError: [Errno 111] Connection refused
```

**解決策**:

1. vLLMサーバーが起動しているか確認:
   ```bash
   curl http://localhost:8000/v1/models
   ```

2. サーバーを起動:
   ```bash
   uv run python src/start_vllm_server.py --config config/generation_config.yaml
   ```

3. ポートが使用中か確認:
   ```bash
   lsof -i :8000
   ```

### 症状2: モデルが見つからない

```
Error code: 404 - {'error': {'message': 'The model `default` does not exist.', 'type': 'NotFoundError'}}
```

**解決策**:

`src/generate_dataset.py` でモデル名をフルパスに変更:

```python
model_name = "models/Qwen2.5-32B-Instruct-bnb-4bit"
```

vLLMは `model="default"` をサポートしていない。

---

## データ生成エラー

### 症状1: JSON解析エラー

```
json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**原因**:
- モデルの出力が空
- JSON形式ではない
- マークダウンコードブロックで囲まれている

**解決策**:

`src/generate_dataset.py` のJSON抽出ロジックを確認:

```python
# ```json ... ``` で囲まれている場合を処理
if "```json" in generated_text:
    json_start = generated_text.find("```json") + 7
    json_end = generated_text.find("```", json_start)
    json_str = generated_text[json_start:json_end].strip()
elif "[" in generated_text:
    json_start = generated_text.find("[")
    json_end = generated_text.rfind("]") + 1
    json_str = generated_text[json_start:json_end]
```

`max_tokens` を増やす:

```yaml
generation:
  max_new_tokens: 2000  # 増やす
```

### 症状2: 「あーし」が含まれない

**原因**:
- プロンプトの例が少ない
- 「あーし」使用の強調が弱い

**解決策**:

`prompts/system_prompt.txt` を改善:

1. 例を増やす（3個 → 8個）
2. 「**重要:** Gyaruの発言では、一人称「あーし」を積極的に使用してください」を追加
3. 例の多くに「あーし」を含める

---

## よくあるエラーと解決策

### エラー1: uvが見つからない

```
shutil.which("uv") returns None
```

**解決策**:

```bash
# uvをインストール
curl -LsSf https://astral.sh/uv/install.sh | sh
```

または、`src/start_vllm_server.py` のフォールバックロジックを使用（`sys.executable`）。

### エラー2: WSL2でpin_memory警告

```
WARNING: Using 'pin_memory=False' as WSL is detected. This may slow down the performance.
```

**解決策**: これは警告であり、エラーではない。WSL2環境では正常動作。無視して問題なし。

### エラー3: モデルダウンロードが遅い

**解決策**:

```bash
# Hugging Face CLIでダウンロード
huggingface-cli download JunHowie/Qwen3-32B-GPTQ-Int8 \
  --local-dir models/Qwen3-32B-GPTQ-Int8

# またはPythonスクリプト
uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='JunHowie/Qwen3-32B-GPTQ-Int8',
    local_dir='models/Qwen3-32B-GPTQ-Int8'
)
"
```

### エラー4: サーバーが応答しない

**症状**: リクエストがタイムアウトする

**解決策**:

1. サーバーログを確認:
   ```bash
   tail -f /tmp/vllm_server_*.log
   ```

2. GPU使用率を確認:
   ```bash
   nvidia-smi
   ```

3. サーバーを再起動:
   ```bash
   pkill -9 -f "vllm"
   uv run python src/start_vllm_server.py --config config/generation_config.yaml
   ```

---

## デバッグ手順

### 1. サーバー状態確認

```bash
# サーバーが起動しているか
ps aux | grep vllm

# APIが応答するか
curl http://localhost:8000/v1/models

# GPU使用状況
nvidia-smi
```

### 2. ログ確認

```bash
# サーバーログ
tail -f /tmp/vllm_server_*.log

# 生成ログ
tail -f /tmp/dataset_generation_*.log
```

### 3. 設定確認

```bash
# 設定ファイルの構文チェック
python3 -c "import yaml; yaml.safe_load(open('config/generation_config.yaml'))"
```

### 4. モデル確認

```bash
# モデルパスの存在確認
ls -la models/Qwen2.5-32B-Instruct-bnb-4bit

# モデルサイズ確認
du -sh models/Qwen2.5-32B-Instruct-bnb-4bit
```

---

## 予防策

1. **VRAM管理**: 4bit量子化モデルを使用
2. **設定の確認**: 起動前に `config/generation_config.yaml` を確認
3. **ログの確認**: エラー発生時はログを確認
4. **段階的テスト**: 1バッチずつテストしてから全バッチ実行
5. **バックアップ**: 生成データは `outputs/raw/` に保存される

---

## サポート

問題が解決しない場合:

1. [AGENTS.md](../AGENTS.md) を確認
2. [IMPLEMENTATION.md](IMPLEMENTATION.md) を確認
3. vLLMのドキュメント: https://docs.vllm.ai/
4. ログファイルを確認してエラーメッセージを特定
