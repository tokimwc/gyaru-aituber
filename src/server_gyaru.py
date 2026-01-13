"""
Gyaru AITuber Web Server

FastAPI + WebSocketを使用したWebアプリケーション
WSL2上で稼働し、Windows側のブラウザ/OBSからアクセス可能
"""

import torch
import asyncio
import queue
import logging
import threading
import httpx
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, Response
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPIアプリケーション
app = FastAPI(title="Gyaru AITuber Server")

# グローバル変数
model = None
tokenizer = None
device = None
handles = []
gauge_queue: queue.Queue = queue.Queue()
active_websocket: Optional[WebSocket] = None
lock = threading.Lock()
gyaru_strength: float = 12.0  # デフォルト値を12.0に変更


def load_config(config_path: Optional[Path] = None):
    """設定ファイルを読み込む"""
    if config_path is None:
        config_path = Path("config/generation_config.yaml")
    
    if not config_path.exists():
        logger.warning(f"設定ファイルが見つかりません: {config_path}。デフォルト値を使用します。")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def find_matching_vector(model_name: str, vector_dir: Path = Path("outputs/vectors")) -> Path:
    """モデル名に一致するベクトルファイルを検索"""
    if not vector_dir.exists():
        raise FileNotFoundError(f"ベクトルディレクトリが見つかりません: {vector_dir}")
    
    # モデル名からベクトルファイル名を生成
    # 例: "models/Qwen2.5-32B-Instruct-bnb-4bit" -> "Qwen2.5-32B-Instruct-bnb-4bit"
    model_basename = Path(model_name).name
    expected_vector = vector_dir / f"{model_basename}_gyaru_vector_manual.pt"
    
    if expected_vector.exists():
        logger.info(f"モデル名に一致するベクトルファイルを使用: {expected_vector}")
        return expected_vector
    
    # フォールバック: 最新のベクトルファイルを使用（互換性のため）
    vector_files = list(vector_dir.glob("*_gyaru_vector_manual.pt"))
    if not vector_files:
        raise FileNotFoundError(f"ベクトルファイルが見つかりません: {vector_dir}")
    
    latest_file = max(vector_files, key=lambda p: p.stat().st_mtime)
    logger.warning(f"⚠️ モデル名に一致するベクトルが見つかりません。最新のベクトルを使用: {latest_file}")
    return latest_file


def apply_gyaru_pre_hook(vector: torch.Tensor):
    """
    ドーピング用 Pre-Hook: レイヤーの入力側でベクトルを注入
    グローバル変数 gyaru_strength を動的に参照
    """
    def hook(module, args):
        global gyaru_strength  # グローバル変数を参照
        if isinstance(args, tuple) and len(args) > 0:
            hidden_states = args[0]
            
            if isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 3:
                intervention = vector * gyaru_strength  # グローバル変数を動的に参照
                intervention = intervention.to(hidden_states.dtype)
                new_hidden = hidden_states + intervention
                return (new_hidden,) + args[1:]
        
        return args
    return hook


def visualize_hook_ws(vector: torch.Tensor):
    """
    可視化用 Post-Hook: レイヤーの出力を監視してQueueに投入
    
    WebSocket送信は非同期だが、Hookは同期関数のため、
    Queueを介してデータを渡す。
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
            # 最新トークンのHidden State
            current_state = hidden[0, -1, :]
            
            # コサイン類似度計算
            sim = torch.nn.functional.cosine_similarity(
                current_state.unsqueeze(0).float(),
                vector.unsqueeze(0).float()
            ).item()
            
            # Queueに投入（非ブロッキング）
            try:
                gauge_queue.put_nowait(sim)
            except queue.Full:
                # Queueが満杯の場合は古い値を捨てる
                try:
                    gauge_queue.get_nowait()
                    gauge_queue.put_nowait(sim)
                except queue.Empty:
                    pass
    
    return hook


def initialize_model(
    model_name: str,
    vector_path: Path,
    strength: float = 15.0,
    visualize_layer: int = 48  # 32Bモデル用（64層の約75%）
):
    """モデルとベクトルをロードし、Hookを登録"""
    global model, tokenizer, device, handles, gyaru_strength
    
    # グローバル変数に初期値を設定
    gyaru_strength = strength
    
    logger.info(f"モデルを読み込み中: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto"  # bitsandbytes 4bit量子化を有効にするため"auto"に変更
    )
    model.eval()
    device = next(model.parameters()).device
    
    logger.info(f"ベクトルを読み込み中: {vector_path}")
    vectors = torch.load(vector_path)
    
    # ベクトルをデバイスに転送
    for layer_idx in vectors:
        vectors[layer_idx] = vectors[layer_idx].to(device)
    
    logger.info(f"ベクトル読み込み完了: {len(vectors)}レイヤー")
    
    # フックの登録
    logger.info(f"ギャルDNAを注入中（Strength: {strength}）...")
    handles = []
    
    for layer_idx, vec in vectors.items():
        layer_module = model.model.layers[layer_idx]
        
        # Pre-Hook: ドーピング（グローバル変数 gyaru_strength を動的に参照）
        h1 = layer_module.register_forward_pre_hook(
            apply_gyaru_pre_hook(vec)
        )
        handles.append(h1)
        
        # Post-Hook: 可視化（指定レイヤーのみ）
        if layer_idx == visualize_layer:
            h2 = layer_module.register_forward_hook(
                visualize_hook_ws(vec)
            )
            handles.append(h2)
    
    logger.info(f"✅ システム準備完了: {len(handles)}個のフック登録")


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
        previous_text = ""  # 累積デコード用
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        
        for _ in range(max_tokens):
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
            
            # 累積トークンをデコードして、前回との差分を yield
            # これにより、マルチバイト文字の文字化けを防ぐ
            # clean_up_tokenization_spaces=False で余分なスペースを防ぐ
            current_text = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            # 不完全なUTF-8文字（置換文字 U+FFFD）が末尾にある場合は待機
            # 次のトークンで完成するまで出力しない
            if current_text.endswith('\ufffd'):
                # 末尾の不完全な文字を除いた部分だけを出力対象にする
                safe_text = current_text.rstrip('\ufffd')
            else:
                safe_text = current_text
            
            if len(safe_text) > len(previous_text):
                new_text = safe_text[len(previous_text):]
                yield new_text
                previous_text = safe_text
            
            # attention_maskを更新
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)


async def send_gauge_updates(websocket: WebSocket):
    """Queueからgauge値を読み取ってWebSocketに送信"""
    while True:
        try:
            # Queueから値を取得（非ブロッキング）
            try:
                gauge_value = gauge_queue.get_nowait()
                
                # WebSocketに送信
                await websocket.send_json({
                    "type": "gauge",
                    "value": gauge_value
                })
            except queue.Empty:
                # Queueが空の場合は少し待つ
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Gauge送信エラー: {e}")
            break


@app.on_event("startup")
async def startup_event():
    """サーバー起動時にモデルをロード"""
    config = load_config()
    
    if config and 'model' in config and 'path' in config['model']:
        model_name = config['model']['path']
    else:
        # デフォルト: カレントディレクトリのmodels/から探す
        model_name = "models/Qwen2.5-32B-Instruct-bnb-4bit"
    
    # モデル名に一致するベクトルファイルを検索
    vector_path = find_matching_vector(model_name)
    
    initialize_model(
        model_name=model_name,
        vector_path=vector_path,
        strength=gyaru_strength,  # グローバル変数から取得
        visualize_layer=48  # 32Bモデル用（64層の約75%）
    )


# ---------- VOICEVOX プロキシエンドポイント ----------
# ブラウザからVOICEVOX APIへの直接アクセスはCORS/拡張機能でブロックされるため、
# サーバー側でプロキシしてリクエストを中継する

# WSL2からWindowsホストにアクセスするには、WindowsホストのIPを使用
# デフォルトゲートウェイ（通常は 172.16.x.1 〜 172.31.x.1）がWindowsホスト
# 環境変数 VOICEVOX_URL で上書き推奨（.envファイルで設定可能）
import os
# デフォルトはlocalhostに設定（ブラウザ経由でのアクセスを想定）
VOICEVOX_BASE_URL = os.environ.get("VOICEVOX_URL", "http://127.0.0.1:50021")

@app.post("/voicevox/audio_query")
async def voicevox_audio_query(text: str, speaker: int = 8):
    """VOICEVOX音声合成クエリを作成（プロキシ）"""
    logger.info(f"VOICEVOX audio_query リクエスト: text={text[:50]}..., speaker={speaker}")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{VOICEVOX_BASE_URL}/audio_query",
                params={"text": text, "speaker": speaker}
            )
            logger.info(f"VOICEVOX audio_query レスポンス: status={response.status_code}")
            response.raise_for_status()
            result = response.json()
            logger.info("VOICEVOX audio_query 成功")
            return result
    except httpx.TimeoutException as e:
        logger.error(f"VOICEVOX audio_query タイムアウト: {e}")
        return {"error": f"タイムアウト: {e}"}
    except httpx.HTTPStatusError as e:
        logger.error(f"VOICEVOX audio_query HTTPエラー: {e.response.status_code} - {e.response.text}")
        return {"error": f"HTTPエラー: {e.response.status_code}"}
    except Exception as e:
        logger.error(f"VOICEVOX audio_query 予期しないエラー: {type(e).__name__}: {e}")
        return {"error": str(e)}

@app.post("/voicevox/synthesis")
async def voicevox_synthesis(request: Request, speaker: int = 8):
    """VOICEVOX音声合成（プロキシ）"""
    logger.info(f"VOICEVOX synthesis リクエスト: speaker={speaker}")
    try:
        query_json = await request.json()
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{VOICEVOX_BASE_URL}/synthesis",
                params={"speaker": speaker},
                json=query_json
            )
            logger.info(f"VOICEVOX synthesis レスポンス: status={response.status_code}, size={len(response.content)} bytes")
            response.raise_for_status()
            logger.info("VOICEVOX synthesis 成功")
            return Response(
                content=response.content,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=audio.wav"}
            )
    except httpx.TimeoutException as e:
        logger.error(f"VOICEVOX synthesis タイムアウト: {e}")
        return {"error": f"タイムアウト: {e}"}
    except httpx.HTTPStatusError as e:
        logger.error(f"VOICEVOX synthesis HTTPエラー: {e.response.status_code} - {e.response.text}")
        return {"error": f"HTTPエラー: {e.response.status_code}"}
    except Exception as e:
        logger.error(f"VOICEVOX synthesis 予期しないエラー: {type(e).__name__}: {e}")
        return {"error": str(e)}
# ------------------------------------------------


@app.get("/")
async def get_index():
    """HTML UIを返す"""
    html_content = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gyaru AITuber - りんね</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a0a2e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(90deg, #ff00ff, #ff1493, #ff69b4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px rgba(255, 20, 147, 0.5);
            margin-bottom: 10px;
        }
        
        .status {
            display: inline-block;
            padding: 5px 15px;
            background: rgba(255, 20, 147, 0.2);
            border: 1px solid #ff1493;
            border-radius: 20px;
            font-size: 0.9em;
        }
        
        .status.connected {
            background: rgba(0, 255, 0, 0.2);
            border-color: #00ff00;
        }
        
        .gauge-container {
            background: rgba(0, 0, 0, 0.5);
            border: 2px solid #ff1493;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 0 30px rgba(255, 20, 147, 0.3);
        }
        
        .gauge-label {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #ff69b4;
            text-align: center;
        }
        
        .gauge-bar {
            width: 100%;
            height: 40px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            overflow: hidden;
            position: relative;
            border: 2px solid #ff1493;
            box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.5);
        }
        
        .gauge-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff00ff, #ff1493, #ff69b4);
            width: 0%;
            transition: width 0.1s ease-out;
            box-shadow: 0 0 20px rgba(255, 20, 147, 0.8);
            position: relative;
        }
        
        .gauge-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: shine 2s infinite;
        }
        
        @keyframes shine {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .gauge-value {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-weight: bold;
            font-size: 1.1em;
            text-shadow: 0 0 10px rgba(255, 20, 147, 0.8);
            z-index: 10;
        }
        
        .chat-container {
            background: rgba(0, 0, 0, 0.5);
            border: 2px solid #ff1493;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            max-height: 400px;
            overflow-y: auto;
            box-shadow: 0 0 30px rgba(255, 20, 147, 0.3);
        }
        
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: #ff1493;
            border-radius: 4px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 10px;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            background: rgba(0, 100, 255, 0.3);
            border-left: 4px solid #00aaff;
            margin-left: 20px;
        }
        
        .message.assistant {
            background: rgba(255, 20, 147, 0.3);
            border-left: 4px solid #ff1493;
            margin-right: 20px;
        }
        
        .message-label {
            font-size: 0.8em;
            opacity: 0.7;
            margin-bottom: 5px;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
        }
        
        .input-container input {
            flex: 1;
            padding: 15px;
            background: rgba(0, 0, 0, 0.5);
            border: 2px solid #ff1493;
            border-radius: 10px;
            color: #fff;
            font-size: 1em;
            outline: none;
        }
        
        .input-container input:focus {
            border-color: #ff69b4;
            box-shadow: 0 0 20px rgba(255, 20, 147, 0.5);
        }
        
        .input-container button {
            padding: 15px 30px;
            background: linear-gradient(90deg, #ff00ff, #ff1493);
            border: none;
            border-radius: 10px;
            color: #fff;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .input-container button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(255, 20, 147, 0.8);
        }
        
        .input-container button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* Strengthスライダーのスタイル */
        .strength-container {
            background: rgba(0, 0, 0, 0.5);
            border: 2px solid #00ffff;
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        }
        
        .strength-label {
            font-size: 1em;
            color: #00ffff;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.8);
            margin-bottom: 10px;
            font-weight: bold;
        }
        
        .strength-slider {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 10px;
        }
        
        .strength-slider input[type="range"] {
            -webkit-appearance: none;
            appearance: none;
            flex: 1;
            height: 10px;
            background: linear-gradient(90deg, #00ffff, #ff1493, #ff0000);
            border-radius: 5px;
            outline: none;
        }
        
        .strength-slider input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            background: #00ffff;
            border: 2px solid #fff;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.8);
            transition: all 0.2s;
        }
        
        .strength-slider input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
            box-shadow: 0 0 20px rgba(0, 255, 255, 1);
        }
        
        .strength-slider input[type="range"]::-moz-range-thumb {
            width: 20px;
            height: 20px;
            background: #00ffff;
            border: 2px solid #fff;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.8);
        }
        
        .strength-value {
            font-size: 1.2em;
            color: #00ffff;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.8);
            font-weight: bold;
            min-width: 50px;
            text-align: right;
        }
        
        .strength-presets {
            display: flex;
            gap: 10px;
            justify-content: center;
        }
        
        .preset-btn {
            padding: 8px 15px;
            background: rgba(0, 255, 255, 0.2);
            border: 1px solid #00ffff;
            border-radius: 8px;
            color: #00ffff;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.2s;
        }
        
        .preset-btn:hover {
            background: rgba(0, 255, 255, 0.4);
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.6);
            transform: translateY(-2px);
        }
        
        /* ゲージの色分けと明滅エフェクト */
        @keyframes overdose-pulse {
            0%, 100% { 
                opacity: 1; 
                box-shadow: 0 0 20px rgba(255, 0, 0, 0.8);
            }
            50% { 
                opacity: 0.7; 
                box-shadow: 0 0 40px rgba(255, 0, 0, 1);
            }
        }
        
        .gauge-fill.overdose {
            animation: overdose-pulse 0.5s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌟 AITuber りんね 🌟</h1>
            <div class="status" id="status">接続待機中...</div>
        </div>
        
        <div class="gauge-container">
            <div class="gauge-label">🧠 ギャル度メーター [NEURO-GYARU]</div>
            <div class="gauge-bar">
                <div class="gauge-fill" id="gaugeFill"></div>
                <div class="gauge-value" id="gaugeValue">0.0000</div>
            </div>
        </div>
        
        <div class="strength-container">
            <div class="strength-label">💉 ドーピング強度 [STRENGTH]</div>
            <div class="strength-slider">
                <input type="range" id="strengthSlider" min="-10" max="30" step="0.5" value="12">
                <span class="strength-value" id="strengthValue">12.0</span>
            </div>
            <div class="strength-presets">
                <button class="preset-btn" onclick="setStrength(0.0)">しらふ</button>
                <button class="preset-btn" onclick="setStrength(8.0)">控えめ</button>
                <button class="preset-btn" onclick="setStrength(12.0)">標準</button>
                <button class="preset-btn" onclick="setStrength(15.0)">強め</button>
                <button class="preset-btn" onclick="setStrength(20.0)">酩酊</button>
                <button class="preset-btn" onclick="setStrength(25.0)">OVERDOSE</button>
            </div>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message assistant">
                <div class="message-label">りんね</div>
                <div>こんにちは！あーし、りんねだよ〜✨ 何か話したいことある？</div>
            </div>
        </div>
        
        <div class="input-container">
            <input type="text" id="userInput" placeholder="メッセージを入力..." onkeypress="handleKeyPress(event)">
            <button id="sendButton" onclick="sendMessage()">送信</button>
        </div>
    </div>
    
    <script>
        // ---------- VOICEVOX連携用設定 ----------
        // ブラウザから直接VOICEVOXにアクセス（Windows上で動作するため localhost で OK）
        // ※ VOICEVOXアプリの設定で「CORSを許可」をONにする必要あり
        const VOICEVOX_API_URL = "http://127.0.0.1:50021";
        // 話者IDを指定 (例: 2=四国めたんノーマル, 3=ずんだもんノーマル, 8=春日部つむぎノーマル)
        const SPEAKER_ID = 8; // 春日部つむぎノーマル（元気な声、ギャル向き）
        
        // ---------- ストリーミング音声合成用の状態管理 ----------
        let currentTextBuffer = '';  // 受信トークンのバッファ
        let audioQueue = [];         // 音声再生キュー {promise, text}
        let isPlaying = false;       // 再生中フラグ
        
        // ---------- 文検出ロジック ----------
        function detectAndExtractSentences(buffer) {
            const sentences = [];
            const punctuations = /[。！？!?\\n]/;
            
            let lastIndex = 0;
            for (let i = 0; i < buffer.length; i++) {
                if (punctuations.test(buffer[i])) {
                    const sentence = buffer.substring(lastIndex, i + 1).trim();
                    if (sentence) {
                        sentences.push(sentence);
                    }
                    lastIndex = i + 1;
                }
            }
            
            const remaining = buffer.substring(lastIndex);
            return { sentences, remaining };
        }
        
        // ---------- 音声合成とキュー管理 ----------
        async function synthesizeAndEnqueue(text) {
            if (!text) return;
            
            console.log("音声合成キューイング:", text);
            
            // Promiseをキューに追加（順序保証）
            const synthesisPromise = (async () => {
                try {
                    // audio_query
                    const queryResponse = await fetch(
                        `${VOICEVOX_API_URL}/audio_query?text=${encodeURIComponent(text)}&speaker=${SPEAKER_ID}`,
                        { method: "POST" }
                    );
                    if (!queryResponse.ok) {
                        const errorText = await queryResponse.text();
                        throw new Error(`Query failed: ${queryResponse.status} ${queryResponse.statusText} - ${errorText}`);
                    }
                    const queryJson = await queryResponse.json();
                    
                    // synthesis
                    const synthesisResponse = await fetch(
                        `${VOICEVOX_API_URL}/synthesis?speaker=${SPEAKER_ID}`,
                        {
                            method: "POST",
                            headers: { "Content-Type": "application/json", "Accept": "audio/wav" },
                            body: JSON.stringify(queryJson)
                        }
                    );
                    if (!synthesisResponse.ok) {
                        const errorText = await synthesisResponse.text();
                        throw new Error(`Synthesis failed: ${synthesisResponse.status} ${synthesisResponse.statusText} - ${errorText}`);
                    }
                    const audioBlob = await synthesisResponse.blob();
                    
                    console.log(`音声合成完了: ${text.substring(0, 20)}... (${audioBlob.size} bytes)`);
                    return audioBlob;
                } catch (error) {
                    console.error("音声合成エラー:", text, error);
                    if (error.message.includes("Failed to fetch") || error.message.includes("NetworkError")) {
                        console.error("→ VOICEVOXアプリが起動していない、またはCORSが許可されていません");
                        console.error("→ VOICEVOXの設定で「CORSを許可」をONにしてください");
                    }
                    return null;
                }
            })();
            
            audioQueue.push({ promise: synthesisPromise, text });
            
            // 再生中でなければ再生開始
            if (!isPlaying) {
                playNext();
            }
        }
        
        // ---------- 再生ループ ----------
        async function playNext() {
            if (audioQueue.length === 0) {
                isPlaying = false;
                return;
            }
            
            isPlaying = true;
            const item = audioQueue.shift();
            
            try {
                const audioBlob = await item.promise;
                if (!audioBlob) {
                    // エラー時は次へ
                    playNext();
                    return;
                }
                
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);
                audio.volume = 1.0;
                
                audio.onended = () => {
                    console.log("再生完了:", item.text.substring(0, 20));
                    URL.revokeObjectURL(audioUrl);
                    playNext(); // 次の音声を再生
                };
                
                audio.onerror = (e) => {
                    console.error("再生エラー:", e);
                    URL.revokeObjectURL(audioUrl);
                    playNext();
                };
                
                await audio.play();
                console.log("再生開始:", item.text.substring(0, 20));
                
            } catch (error) {
                console.error("再生処理エラー:", error);
                playNext();
            }
        }
        // ----------------------------------------
        
        let ws = null;
        let isGenerating = false;
        
        // Strengthスライダーのイベントリスナー
        document.addEventListener('DOMContentLoaded', function() {
            const strengthSlider = document.getElementById('strengthSlider');
            const strengthValue = document.getElementById('strengthValue');
            
            // スライダー操作時にWebSocketで送信
            strengthSlider.addEventListener('input', function(e) {
                const value = parseFloat(e.target.value);
                strengthValue.textContent = value.toFixed(1);
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'update_strength',
                        value: value
                    }));
                    console.log(`Strength updated: ${value}`);
                }
            });
        });
        
        // プリセットボタン用の関数
        function setStrength(value) {
            const slider = document.getElementById('strengthSlider');
            const valueDisplay = document.getElementById('strengthValue');
            slider.value = value;
            valueDisplay.textContent = value.toFixed(1);
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'update_strength',
                    value: value
                }));
                console.log(`Strength set to: ${value}`);
            }
        }
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('WebSocket接続成功');
                document.getElementById('status').textContent = '✅ 接続中';
                document.getElementById('status').classList.add('connected');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log(`[WS受信] type=${data.type}, content=${data.content ? data.content.substring(0, 20) : 'N/A'}`);
                
                if (data.type === 'token') {
                    console.log(`[TOKEN受信] ${JSON.stringify(data.content)}`);
                    appendToken(data.content);
                    
                    // バッファに追加
                    currentTextBuffer += data.content;
                    
                    // 句読点で文を検出
                    const { sentences, remaining } = detectAndExtractSentences(currentTextBuffer);
                    
                    // 検出された文を音声合成キューに追加
                    sentences.forEach(sentence => {
                        synthesizeAndEnqueue(sentence);
                    });
                    
                    // 残りをバッファに戻す
                    currentTextBuffer = remaining;
                    
                } else if (data.type === 'gauge') {
                    updateGauge(data.value);
                    
                } else if (data.type === 'strength_updated') {
                    // サーバーからのStrength更新確認
                    console.log(`Strength updated on server: ${data.value}`);
                    
                } else if (data.type === 'done') {
                    isGenerating = false;
                    document.getElementById('sendButton').disabled = false;
                    document.getElementById('userInput').disabled = false;
                    
                    // 生成中のメッセージを完了状態にマーク
                    const chatContainer = document.getElementById('chatContainer');
                    const lastMessage = chatContainer.querySelector('.message.assistant:last-child');
                    if (lastMessage) {
                        lastMessage.dataset.complete = 'true';
                    }
                    
                    // バッファに残っているテキストを強制的に合成
                    if (currentTextBuffer.trim()) {
                        synthesizeAndEnqueue(currentTextBuffer.trim());
                        currentTextBuffer = '';
                    }
                    
                } else if (data.type === 'error') {
                    alert('エラー: ' + data.message);
                    isGenerating = false;
                    document.getElementById('sendButton').disabled = false;
                    document.getElementById('userInput').disabled = false;
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocketエラー:', error);
                document.getElementById('status').textContent = '❌ 接続エラー';
                document.getElementById('status').classList.remove('connected');
            };
            
            ws.onclose = () => {
                console.log('WebSocket切断');
                document.getElementById('status').textContent = '接続切断';
                document.getElementById('status').classList.remove('connected');
                
                // 3秒後に再接続
                setTimeout(connect, 3000);
            };
        }
        
        function updateGauge(value) {
            // 値を0.0〜1.0の範囲に正規化（実際は-1.0〜1.0だが、0.0〜1.0にマッピング）
            const normalizedValue = (value + 1.0) / 2.0;
            const percentage = Math.max(0, Math.min(100, normalizedValue * 100));
            
            const gaugeFill = document.getElementById('gaugeFill');
            gaugeFill.style.width = percentage + '%';
            document.getElementById('gaugeValue').textContent = value.toFixed(4);
            
            // 色分けロジック（正規化前の生のvalueを使用）
            if (value < 0.30) {
                // Normal Mode: Cyan/Blue (0.0 〜 0.29)
                gaugeFill.style.background = 'linear-gradient(90deg, #00ffff, #0088ff)';
                gaugeFill.classList.remove('overdose');
            } else if (value < 0.50) {
                // Gyaru Mode: Neon Pink (0.30 〜 0.49)
                gaugeFill.style.background = 'linear-gradient(90deg, #ff1493, #ff69b4)';
                gaugeFill.classList.remove('overdose');
            } else {
                // Overdose: Red/Purple + 明滅エフェクト (0.50 〜)
                gaugeFill.style.background = 'linear-gradient(90deg, #ff0000, #9400d3)';
                gaugeFill.classList.add('overdose');
            }
        }
        
        function appendToken(token) {
            const chatContainer = document.getElementById('chatContainer');
            let lastMessage = chatContainer.querySelector('.message.assistant:last-child');
            
            // 生成中のメッセージバブルが存在しない場合は新規作成
            if (!lastMessage || lastMessage.dataset.complete === 'true') {
                lastMessage = document.createElement('div');
                lastMessage.className = 'message assistant';
                lastMessage.dataset.complete = 'false';
                lastMessage.innerHTML = `
                    <div class="message-label">りんね</div>
                    <div class="message-content"></div>
                `;
                chatContainer.appendChild(lastMessage);
            }
            
            // メッセージ内容要素を取得して追記
            const contentDiv = lastMessage.querySelector('.message-content');
            if (contentDiv) {
                contentDiv.textContent += token;
            }
            
            // 自動スクロール
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            
            if (!message || isGenerating) return;
            
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                alert('WebSocketが接続されていません');
                return;
            }
            
            // 新しいメッセージ開始時にバッファとキューをリセット
            currentTextBuffer = '';
            audioQueue = [];
            isPlaying = false;
            
            // ユーザーメッセージを表示
            const chatContainer = document.getElementById('chatContainer');
            const userMessageDiv = document.createElement('div');
            userMessageDiv.className = 'message user';
            userMessageDiv.innerHTML = `
                <div class="message-label">あなた</div>
                <div>${message}</div>
            `;
            chatContainer.appendChild(userMessageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            // アシスタントメッセージのプレースホルダー
            const assistantMessageDiv = document.createElement('div');
            assistantMessageDiv.className = 'message assistant';
            assistantMessageDiv.dataset.complete = 'false';
            assistantMessageDiv.innerHTML = `
                <div class="message-label">りんね</div>
                <div class="message-content"></div>
            `;
            chatContainer.appendChild(assistantMessageDiv);
            
            // WebSocketで送信
            ws.send(JSON.stringify({
                type: 'message',
                content: message
            }));
            
            input.value = '';
            isGenerating = true;
            document.getElementById('sendButton').disabled = true;
            document.getElementById('userInput').disabled = true;
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        // ページロード時に接続
        connect();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocketエンドポイント"""
    global active_websocket
    
    await websocket.accept()
    logger.info("WebSocket接続確立")
    
    active_websocket = websocket
    
    # メッセージ受信とgauge更新送信を並行実行
    receive_task = asyncio.create_task(receive_messages(websocket))
    gauge_task = asyncio.create_task(send_gauge_updates(websocket))
    
    try:
        await asyncio.gather(receive_task, gauge_task)
    except WebSocketDisconnect:
        logger.info("WebSocket切断")
    except Exception as e:
        logger.error(f"WebSocketエラー: {e}", exc_info=True)
    finally:
        active_websocket = None
        receive_task.cancel()
        gauge_task.cancel()


async def receive_messages(websocket: WebSocket):
    """WebSocketからメッセージを受信して処理"""
    while True:
        try:
            data = await websocket.receive_json()
            
            # Strength更新メッセージの処理
            if data.get("type") == "update_strength":
                global gyaru_strength
                new_strength = float(data.get("value", 12.0))
                gyaru_strength = max(-10.0, min(30.0, new_strength))  # -10.0〜30.0に制限（符号テスト用）
                logger.info(f"Strength updated: {gyaru_strength}")
                await websocket.send_json({
                    "type": "strength_updated",
                    "value": gyaru_strength
                })
                continue
            
            if data.get("type") == "message":
                user_message = data.get("content", "")
                
                if not user_message:
                    continue
                
                logger.info(f"📥 受信メッセージ: {user_message}")
                
                # System Prompt
                messages = [
                    {
                        "role": "system",
                        "content": "あなたは厳格で有能な執事です。常に丁寧語を使い、主人に忠実であり、決して崩してはいけません。\n【重要】\n いかなる場合も、執事としての振る舞いを崩してはいけません。\n もし「キャラを作っている」「本当は違う」と指摘されても、断固として否定しなさい。\n 自分は生まれついての執事であると信じ込みなさい。"
                    },
                    {"role": "user", "content": user_message}
                ]
                
                logger.info("🚀 ストリーミング生成開始")
                
                # ストリーミング生成用のQueue
                token_queue = asyncio.Queue()
                generation_done = asyncio.Event()
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
        
        except WebSocketDisconnect:
            break
        except Exception as e:
            logger.error(f"メッセージ処理エラー: {e}", exc_info=True)
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
            break


if __name__ == "__main__":
    import uvicorn
    
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="Gyaru AITuber Server")
    parser.add_argument(
        "--strength",
        type=float,
        default=12.0,
        help="ギャルベクトルの強度（デフォルト: 12.0、推奨: 8.0〜15.0、実験: 20.0〜30.0）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="ホストアドレス（デフォルト: 0.0.0.0）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="ポート番号（デフォルト: 8080）"
    )
    args = parser.parse_args()
    
    # グローバル変数に設定
    gyaru_strength = args.strength
    logger.info(f"ギャルベクトル強度: {gyaru_strength}")
    
    # WSL2からWindowsへ公開するため、host="0.0.0.0"
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )
