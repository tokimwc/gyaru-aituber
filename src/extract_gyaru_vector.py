"""
Gyaru Vector Extraction Script

RepE (Representation Engineering) を使用して、
Qwen2.5-32B-Instruct-bnb-4bit モデルから「ギャル成分」を表す制御ベクトルを抽出する。
"""

import json
import logging
import torch
from pathlib import Path
from typing import List, Optional
import argparse
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry

# --- 🔥 モンキーパッチ開始 🔥 ---
# repengのControlModuleが元のモデル属性（attention_type等）を
# 通過させるように無理やり機能を拡張します。
from repeng.control import ControlModule

def getattr_monkey_patch(self, name):
    """ControlModule自身が持っていない属性は、ラップしているmoduleから探す"""
    try:
        # まず自分自身の属性を確認
        return object.__getattribute__(self, name)
    except AttributeError:
        # なければラップしているmoduleから取得
        module = object.__getattribute__(self, "module")
        return getattr(module, name)

# クラスにメソッドを注入
ControlModule.__getattr__ = getattr_monkey_patch
# --- 🔥 モンキーパッチ終了 🔥 ---

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
    
    # 最新のファイルを取得
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


def create_dataset_entries(
    raw_data: List[dict],
    tokenizer: AutoTokenizer,
    max_entries: Optional[int] = None
) -> List[DatasetEntry]:
    """データセットをDatasetEntry形式に変換"""
    logger.info("DatasetEntryを作成中...")
    
    # データ数を制限（OOM対策）
    if max_entries:
        raw_data = raw_data[:max_entries]
        logger.info(f"データ数を {max_entries} 件に制限しました")
    
    dataset = []
    
    # ユーザープロンプトのバリエーション
    user_prompts = [
        "その内容について話して。",
        "どう思いますか？",
        "詳しく教えて。",
        "反応してください。",
        "感想を教えて。"
    ]
    
    import random
    
    for i, item in enumerate(raw_data):
        # standardとgyaruの文章を取得
        positive = item["gyaru"]  # ギャル口調
        negative = item["standard"]  # 標準語
        
        # トピックに基づいてユーザープロンプトを作成
        topic = item.get('topic', '雑談')
        user_text = f"トピック: {topic}\nこれについて話して。"
        
        # Chat Templateの適用
        messages = [{"role": "user", "content": user_text}]
        prefix = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        dataset.append(DatasetEntry(
            positive=f"{prefix}{positive}",
            negative=f"{prefix}{negative}"
        ))
        
        if (i + 1) % 20 == 0:
            logger.info(f"  {i + 1}/{len(raw_data)} エントリを作成しました")
    
    logger.info(f"DatasetEntry作成完了: {len(dataset)}エントリ")
    return dataset


def extract_vector(
    model_name: str,
    dataset: List[DatasetEntry],
    target_layers: List[int],
    batch_size: int = 2,
    device_map: str = "auto",
    trust_remote_code: bool = True
) -> ControlVector:
    """制御ベクトルを抽出"""
    logger.info(f"モデルを読み込み中: {model_name}")
    
    # トークナイザー読み込み
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # モデル読み込み（既に4bit量子化されているため、そのままロード）
    logger.info("モデルをロード中（既に4bit量子化済み）...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=trust_remote_code
        # モデルが既に4bit量子化されているため、load_in_4bitは不要
    )
    
    # RepE用にモデルをラップ（対象レイヤーのみ）
    logger.info(f"ControlModelにラップ中（レイヤー: {target_layers[0]}-{target_layers[-1]}）...")
    model = ControlModel(model, target_layers)
    
    # ベクトル抽出
    logger.info(f"ベクトル抽出を開始（バッチサイズ: {batch_size}）...")
    vector = ControlVector.train(
        model,
        tokenizer,
        dataset,
        batch_size=batch_size
    )
    
    logger.info("ベクトル抽出完了")
    return vector


def save_vector(vector: ControlVector, output_dir: Path, model_name: str):
    """ベクトルを保存"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GGUF形式で保存
    gguf_path = output_dir / f"{model_name.replace('/', '_').replace('-', '_')}_gyaru_vector.gguf"
    logger.info(f"GGUF形式で保存中: {gguf_path}")
    try:
        vector.export_gguf(str(gguf_path))
        logger.info(f"✅ GGUF形式で保存完了: {gguf_path}")
    except Exception as e:
        logger.warning(f"⚠️  GGUF形式での保存に失敗: {e}")
    
    # PyTorch形式で保存
    pt_path = output_dir / "gyaru_vector_obj.pt"
    logger.info(f"PyTorch形式で保存中: {pt_path}")
    try:
        torch.save(vector, pt_path)
        logger.info(f"✅ PyTorch形式で保存完了: {pt_path}")
    except Exception as e:
        logger.warning(f"⚠️  PyTorch形式での保存に失敗: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Gyaru Vector Extraction")
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
        default=None,
        help="使用するデータ数（OOM対策、デフォルト: 全データ）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="バッチサイズ（デフォルト: 2）"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="20-50",
        help="対象レイヤー範囲（例: 20-50）"
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
        model_name = "models/Qwen2.5-32B-Instruct-bnb-4bit"
    
    logger.info(f"使用モデル: {model_name}")
    
    # データセット読み込み
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = find_latest_dataset()
    
    raw_data = load_dataset(dataset_path)
    
    # トークナイザー読み込み（DatasetEntry作成用）
    logger.info("トークナイザーを読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # DatasetEntry作成
    dataset_entries = create_dataset_entries(
        raw_data,
        tokenizer,
        max_entries=args.max_entries
    )
    
    # レイヤー範囲の解析
    layer_start, layer_end = map(int, args.layers.split('-'))
    target_layers = list(range(layer_start, layer_end + 1))
    logger.info(f"対象レイヤー: {target_layers[0]}-{target_layers[-1]} ({len(target_layers)}レイヤー)")
    
    # ベクトル抽出
    vector = extract_vector(
        model_name=model_name,
        dataset=dataset_entries,
        target_layers=target_layers,
        batch_size=args.batch_size,
        trust_remote_code=True
    )
    
    # 保存
    model_name_short = model_name.split('/')[-1]
    save_vector(vector, args.output_dir, model_name_short)
    
    logger.info("🎉 ベクトル抽出完了！")
    logger.info(f"出力ディレクトリ: {args.output_dir}")


if __name__ == "__main__":
    main()
