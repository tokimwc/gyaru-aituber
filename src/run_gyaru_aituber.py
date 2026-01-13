"""
Gyaru AITuber Backend

抽出したギャルベクトルを使用して、リアルタイムで「ギャル人格を憑依させながら、
その憑依度をバーグラフで表示する」AITuberバックエンド

検証結果に基づく修正版:
- Pre-Hookでドーピング（strength=15.0）
- Post-Hookで可視化（読み取り専用）
- System Promptで「あーし」を補強
"""

import torch
import time
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Optional
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def find_latest_vector(vector_dir: Path = Path("outputs/vectors")) -> Path:
    """最新のベクトルファイルを検索"""
    if not vector_dir.exists():
        raise FileNotFoundError(f"ベクトルディレクトリが見つかりません: {vector_dir}")
    
    vector_files = list(vector_dir.glob("*_gyaru_vector_manual.pt"))
    if not vector_files:
        raise FileNotFoundError(f"ベクトルファイルが見つかりません: {vector_dir}")
    
    latest_file = max(vector_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"最新のベクトルファイルを使用: {latest_file}")
    return latest_file


def apply_gyaru_pre_hook(vector: torch.Tensor, strength: float):
    """
    ドーピング用 Pre-Hook: レイヤーの入力側でベクトルを注入
    
    Pre-Hookは入力に介入するため、その後の正規化で減衰する。
    そのため、通常のHookより高いstrength（15.0程度）が必要。
    """
    def hook(module, args):
        # Pre-hookの引数は (input_tensor, ...) のタプル
        if isinstance(args, tuple) and len(args) > 0:
            hidden_states = args[0]
            
            if isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 3:
                # ベクトルを加算 (Broadcasting)
                intervention = vector * strength
                
                # dtypeを合わせる（エラー回避の鍵）
                intervention = intervention.to(hidden_states.dtype)
                
                # 新しいtensorを作成（インプレース操作を避ける）
                new_hidden = hidden_states + intervention
                
                # Pre-hookはタプルで返す必要がある
                return (new_hidden,) + args[1:]
        
        return args
    return hook


def visualize_hook(vector: torch.Tensor, bar_len: int = 20):
    """
    可視化用 Post-Hook: レイヤーの出力を監視（読み取り専用）
    
    読み取り専用なので、タプルを書き換えずに安全に動作する。
    """
    def hook(module, input, output):
        # outputは通常 (hidden_states, past_key_values, ...) のタプル
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
            # 最新トークンのHidden State
            current_state = hidden[0, -1, :]
            
            # コサイン類似度計算（float32で精度確保）
            sim = torch.nn.functional.cosine_similarity(
                current_state.unsqueeze(0).float(),
                vector.unsqueeze(0).float()
            ).item()
            
            # バーグラフ表示
            # Pre-hookの影響で値が小さいかもしれないのでスケーリング
            score = sim * 20
            filled = int(abs(score) * bar_len)
            filled = min(filled, bar_len)
            bar = "▓" * filled + "░" * (bar_len - filled)
            
            # カラー出力
            if score > 0:
                color = "\033[95m"  # Pink (ギャルモード)
                label = "[NEURO-GYARU]"
            else:
                color = "\033[94m"  # Blue (ノーマルモード)
                label = "[NEURO-NORM]"
            
            reset = "\033[0m"
            
            # カーソルを行頭に戻して上書き表示（コンソールが流れるのを防ぐ）
            print(f"{color}{label} {bar} {sim:+.4f}{reset}   \r", end="", flush=True)
    return hook


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Gyaru AITuber Backend")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/generation_config.yaml"),
        help="設定ファイルのパス"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="モデル名（設定ファイルから読み込む場合は省略）"
    )
    parser.add_argument(
        "--vector",
        type=Path,
        default=None,
        help="ベクトルファイルのパス（省略時は最新を使用）"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=15.0,
        help="ギャル度係数（デフォルト: 15.0、Pre-Hookでは高い値が必要）"
    )
    parser.add_argument(
        "--visualize-layer",
        type=int,
        default=48,
        help="可視化対象レイヤー（デフォルト: 48）"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="最大生成トークン数（デフォルト: 128）"
    )
    
    args = parser.parse_args()
    
    # 設定読み込み
    config = load_config(args.config)
    
    # モデル名の決定
    if args.model:
        model_name = args.model
    elif config and 'model' in config and 'path' in config['model']:
        model_name = config['model']['path']
    else:
        # デフォルト: カレントディレクトリのmodels/から探す
        model_name = "models/Qwen2.5-32B-Instruct-bnb-4bit"
    
    logger.info(f"使用モデル: {model_name}")
    
    # ベクトルファイルの決定
    if args.vector:
        vector_path = args.vector
    else:
        vector_path = find_latest_vector()
    
    # モデルとトークナイザー読み込み
    logger.info("モデルを読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.eval()
    
    # ベクトル読み込み
    logger.info(f"ベクトルを読み込み中: {vector_path}")
    vectors = torch.load(vector_path)
    
    # 必要なデバイスに転送
    device = next(model.parameters()).device
    for layer_idx in vectors:
        vectors[layer_idx] = vectors[layer_idx].to(device)
    
    logger.info(f"ベクトル読み込み完了: {len(vectors)}レイヤー")
    
    # フックの登録
    logger.info(f"ギャルDNAを注入中（Strength: {args.strength}）...")
    handles = []
    
    # 抽出されたすべてのレイヤーに適用
    for layer_idx, vec in vectors.items():
        layer_module = model.model.layers[layer_idx]
        
        # 1. 性格改変フック (Pre-Hook: ドーピング)
        h1 = layer_module.register_forward_pre_hook(
            apply_gyaru_pre_hook(vec, args.strength)
        )
        handles.append(h1)
        
        # 2. 監視フック (Post-Hook: 可視化) - 特定レイヤーのみ
        if layer_idx == args.visualize_layer:
            h2 = layer_module.register_forward_hook(
                visualize_hook(vec)
            )
            handles.append(h2)
    
    logger.info(f"✅ システム準備完了。AITuber 'Rinne (Gyaru Ver.)' オンライン。")
    logger.info(f"   ギャル度係数: {args.strength}")
    logger.info(f"   可視化レイヤー: {args.visualize_layer}")
    logger.info("=" * 60)
    
    # インタラクティブ対話ループ
    # 検証結果に基づき、System Promptで「あーし」を補強
    messages = [
        {
            "role": "system",
            "content": "あなたは快活なAITuber「りんね」です。一人称は必ず「あーし」を使ってください。技術的な話も得意ですが、口調は常にギャル語です。"
        }
    ]
    
    try:
        while True:
            try:
                # 入力時は可視化ログと被らないように改行
                user_input = input("\n\nUser: ")
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                
                messages.append({"role": "user", "content": user_input})
                
                # Chat Template適用
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                
                # ストリーミング生成の準備
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                
                generation_kwargs = dict(
                    inputs,
                    streamer=streamer,
                    max_new_tokens=args.max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                # 生成を別スレッドで開始（ストリーミング表示のため）
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                
                print("Rinne: ", end="", flush=True)
                full_response = ""
                
                # ストリーミング中は可視化バーが動く
                for new_text in streamer:
                    print(new_text, end="", flush=True)
                    full_response += new_text
                
                print()  # 改行
                messages.append({"role": "assistant", "content": full_response})
                
            except KeyboardInterrupt:
                print("\n\n中断されました。")
                break
            except Exception as e:
                logger.error(f"エラー発生: {e}", exc_info=True)
                print(f"\nエラー: {e}")
    
    finally:
        # 後始末
        logger.info("フックを解除中...")
        for h in handles:
            h.remove()
        logger.info("切断しました。")


if __name__ == "__main__":
    main()
