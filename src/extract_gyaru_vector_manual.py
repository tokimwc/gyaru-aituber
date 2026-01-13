"""
Gyaru Vector Extraction Script (Manual Implementation)

repengを使わず、PyTorch Hookを直接使用してベクトルを抽出する。
Qwen2.5との互換性問題を回避。
"""

import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import argparse
import yaml
from sklearn.decomposition import PCA

from transformers import AutoModelForCausalLM, AutoTokenizer

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


def find_latest_dataset(dataset_dir: Path = Path("outputs/processed")) -> Path:
    """最新のデータセットファイルを検索"""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"データセットディレクトリが見つかりません: {dataset_dir}")
    
    dataset_files = list(dataset_dir.glob("gyaru_dataset_*.json"))
    if not dataset_files:
        raise FileNotFoundError(f"データセットファイルが見つかりません: {dataset_dir}")
    
    latest_file = max(dataset_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"最新のデータセットファイルを使用: {latest_file}")
    return latest_file


def load_dataset(dataset_path: Path) -> List[dict]:
    """データセットを読み込む"""
    logger.info(f"データセットを読み込み中: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"データセット読み込み完了: {len(data)}ペア")
    return data


def extract_hidden_states_manual(
    model,
    tokenizer,
    dataset: List[dict],
    target_layers: List[int],
    max_entries: Optional[int] = None
) -> Dict[int, np.ndarray]:
    """手動でHidden Statesを抽出してPCAを実行"""
    
    if max_entries:
        dataset = dataset[:max_entries]
        logger.info(f"データ数を {max_entries} 件に制限しました")
    
    # 差分データを格納
    diffs = {layer: [] for layer in target_layers}
    
    logger.info(f"Hidden States抽出を開始（{len(dataset)}ペア）...")
    
    def get_hook(storage_list):
        """Hook関数: 特定の層の出力を取得"""
        def hook(module, input, output):
            # outputは通常 (hidden_states, ...) のタプル
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # 最後のトークンのHidden Stateを取得
            # shape: [batch, seq_len, hidden_dim] -> [batch, hidden_dim]
            storage_list.append(hidden[:, -1, :].detach().cpu())
        return hook
    
    # 各データペアに対して推論
    for idx, item in enumerate(dataset):
        if (idx + 1) % 10 == 0:
            logger.info(f"  {idx + 1}/{len(dataset)} ペアを処理中...")
        
        # プロンプト作成
        user_msg = f"トピック: {item.get('topic', '雑談')}\nこれについて話して。"
        
        # Standard用の入力
        msgs_std = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": item["standard"]}
        ]
        input_std = tokenizer.apply_chat_template(
            msgs_std,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False
        ).to(model.device)
        
        # Gyaru用の入力
        msgs_gya = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": item["gyaru"]}
        ]
        input_gya = tokenizer.apply_chat_template(
            msgs_gya,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False
        ).to(model.device)
        
        # Standard Pass
        std_activations = {}
        handles = []
        for layer_idx in target_layers:
            std_activations[layer_idx] = []
            layer_module = model.model.layers[layer_idx]
            handle = layer_module.register_forward_hook(
                get_hook(std_activations[layer_idx])
            )
            handles.append(handle)
        
        with torch.no_grad():
            model(input_std)
        
        # Hook解除
        for h in handles:
            h.remove()
        
        # Gyaru Pass
        gya_activations = {}
        handles = []
        for layer_idx in target_layers:
            gya_activations[layer_idx] = []
            layer_module = model.model.layers[layer_idx]
            handle = layer_module.register_forward_hook(
                get_hook(gya_activations[layer_idx])
            )
            handles.append(handle)
        
        with torch.no_grad():
            model(input_gya)
        
        # Hook解除
        for h in handles:
            h.remove()
        
        # 差分を計算して蓄積
        for layer_idx in target_layers:
            if std_activations[layer_idx] and gya_activations[layer_idx]:
                vec_std = std_activations[layer_idx][0]  # [1, hidden_dim]
                vec_gya = gya_activations[layer_idx][0]
                
                # 差分 = ギャル - 標準
                diff = vec_gya - vec_std
                diffs[layer_idx].append(diff)
    
    logger.info("PCAで主成分を計算中...")
    final_vectors = {}
    
    for layer_idx, diff_list in diffs.items():
        if not diff_list:
            logger.warning(f"レイヤー {layer_idx} のデータが空です。スキップします。")
            continue
        
        # [num_samples, hidden_dim]
        X = torch.cat(diff_list, dim=0).float().numpy()
        
        # PCAで第1主成分を抽出
        pca = PCA(n_components=1)
        pca.fit(X)
        
        # 方向ベクトル
        direction = pca.components_[0]  # [hidden_dim]
        
        # 符号の調整: ギャルデータの平均との内積が正になるようにする
        mean_diff = np.mean(X, axis=0)
        if np.dot(direction, mean_diff) < 0:
            direction = -direction
        
        final_vectors[layer_idx] = torch.tensor(direction, dtype=torch.float16)
        
        logger.info(f"  レイヤー {layer_idx}: 分散説明率 {pca.explained_variance_ratio_[0]:.4f}")
    
    return final_vectors


def save_vectors(vectors: Dict[int, torch.Tensor], output_dir: Path, model_name: str):
    """ベクトルを保存"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # PyTorch形式で保存
    pt_path = output_dir / f"{model_name}_gyaru_vector_manual.pt"
    logger.info(f"PyTorch形式で保存中: {pt_path}")
    torch.save(vectors, pt_path)
    logger.info(f"✅ 保存完了: {pt_path}")
    
    # 統計情報を保存
    stats_path = output_dir / f"{model_name}_gyaru_vector_stats.txt"
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("# Gyaru Vector Statistics\n\n")
        f.write(f"Total layers: {len(vectors)}\n")
        f.write(f"Layer indices: {sorted(vectors.keys())}\n\n")
        for layer_idx in sorted(vectors.keys()):
            vec = vectors[layer_idx]
            f.write(f"Layer {layer_idx}:\n")
            f.write(f"  Shape: {vec.shape}\n")
            f.write(f"  Norm: {vec.norm().item():.4f}\n")
            f.write(f"  Mean: {vec.mean().item():.6f}\n")
            f.write(f"  Std: {vec.std().item():.6f}\n\n")
    
    logger.info(f"✅ 統計情報保存完了: {stats_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Gyaru Vector Extraction (Manual)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/generation_config.yaml"),
        help="設定ファイルのパス"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="データセットファイルのパス（指定しない場合は最新を使用）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="モデル名（設定ファイルから読み込む場合は省略）"
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=50,
        help="使用するデータ数（デフォルト: 50）"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="15-40",
        help="対象レイヤー範囲（例: 15-40、14Bモデル用デフォルト）"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vectors"),
        help="出力ディレクトリ"
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
        model_name = "models/Qwen2.5-14B-Instruct-bnb-4bit"
    
    logger.info(f"使用モデル: {model_name}")
    
    # データセット読み込み
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = find_latest_dataset()
    
    raw_data = load_dataset(dataset_path)
    
    # レイヤー範囲の解析
    layer_start, layer_end = map(int, args.layers.split('-'))
    target_layers = list(range(layer_start, layer_end + 1))
    logger.info(f"対象レイヤー: {target_layers[0]}-{target_layers[-1]} ({len(target_layers)}レイヤー)")
    
    # モデルとトークナイザー読み込み
    logger.info("トークナイザーを読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    logger.info("モデルを読み込み中...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto"  # bitsandbytes 4bit量子化を有効にするため"auto"に変更
    )
    model.eval()
    
    # ベクトル抽出
    vectors = extract_hidden_states_manual(
        model=model,
        tokenizer=tokenizer,
        dataset=raw_data,
        target_layers=target_layers,
        max_entries=args.max_entries
    )
    
    # 保存
    model_name_short = model_name.split('/')[-1]
    save_vectors(vectors, args.output_dir, model_name_short)
    
    logger.info("🎉 ベクトル抽出完了！")
    logger.info(f"出力ディレクトリ: {args.output_dir}")


if __name__ == "__main__":
    main()
