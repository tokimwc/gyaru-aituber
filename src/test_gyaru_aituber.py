"""
AITuberバックエンド自動テストスクリプト

インタラクティブな対話を自動化して、ギャル化の効果を検証する
"""

import torch
import sys
import logging
from pathlib import Path
from typing import List, Dict
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path = Path("config/generation_config.yaml")):
    """設定ファイルを読み込む"""
    if not config_path.exists():
        return None
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_latest_vector(vector_dir: Path = Path("outputs/vectors")) -> Path:
    """最新のベクトルファイルを検索"""
    vector_files = list(vector_dir.glob("*_gyaru_vector_manual.pt"))
    if not vector_files:
        raise FileNotFoundError(f"ベクトルファイルが見つかりません: {vector_dir}")
    return max(vector_files, key=lambda p: p.stat().st_mtime)


def apply_gyaru_hook(layer_idx: int, vector: torch.Tensor, strength: float):
    """ギャルベクトルを注入するHook関数"""
    def hook(module, input, output):
        # インプレース操作を避けるためclone()を使用
        if isinstance(output, tuple):
            hidden = output[0].clone()
            output_rest = output[1:]
        else:
            hidden = output.clone()
            output_rest = ()
        intervention = vector * strength
        # インプレース操作を避ける
        hidden = hidden + intervention
        return (hidden,) + output_rest
    return hook


def run_test(
    model,
    tokenizer,
    vectors: Dict[int, torch.Tensor],
    test_cases: List[Dict[str, str]],
    strength: float = 1.5,
    max_tokens: int = 128
) -> List[Dict[str, str]]:
    """テストケースを実行して結果を返す"""
    device = next(model.parameters()).device
    
    # フック登録
    handles = []
    for layer_idx, vec in vectors.items():
        layer_module = model.model.layers[layer_idx]
        h = layer_module.register_forward_hook(
            apply_gyaru_hook(layer_idx, vec.to(device), strength)
        )
        handles.append(h)
    
    results = []
    
    try:
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"テスト {i}/{len(test_cases)}: {test_case['name']}")
            
            # メッセージ準備
            messages = [
                {"role": "system", "content": "あなたはAITuberのりんねです。雑談配信をしています。"},
                {"role": "user", "content": test_case['input']}
            ]
            
            # Chat Template適用
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # デコード
            generated_text = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # 結果記録
            result = {
                'name': test_case['name'],
                'input': test_case['input'],
                'output': generated_text,
                'expected_keywords': test_case.get('expected_keywords', [])
            }
            results.append(result)
            
            logger.info(f"応答: {generated_text[:100]}...")
            
    finally:
        # フック解除
        for h in handles:
            h.remove()
    
    return results


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AITuberバックエンド自動テスト")
    parser.add_argument("--strength", type=float, default=1.5, help="ギャル度係数")
    parser.add_argument("--max-tokens", type=int, default=128, help="最大生成トークン数")
    args = parser.parse_args()
    
    # 設定読み込み
    config = load_config()
    
    # モデル名決定
    if config and 'model' in config and 'path' in config['model']:
        model_name = config['model']['path']
    else:
        model_name = "models/Qwen2.5-32B-Instruct-bnb-4bit"
    
    logger.info(f"使用モデル: {model_name}")
    
    # ベクトル読み込み
    vector_path = find_latest_vector()
    logger.info(f"ベクトルファイル: {vector_path}")
    vectors = torch.load(vector_path)
    logger.info(f"ベクトル読み込み完了: {len(vectors)}レイヤー")
    
    # モデル読み込み
    logger.info("モデルを読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.eval()
    logger.info("モデル読み込み完了")
    
    # テストケース定義
    test_cases = [
        {
            'name': '挨拶テスト',
            'input': 'こんにちは',
            'expected_keywords': ['あーし', 'おっはー', 'うぃーす', 'よーっす', 'だし', 'じゃん']
        },
        {
            'name': '技術ネタテスト',
            'input': 'Pythonでエラーが出るんだけど',
            'expected_keywords': ['マジ', 'エラー', 'ログ', '見せて', 'だし', 'じゃん']
        },
        {
            'name': '雑談テスト',
            'input': '今日何してた？',
            'expected_keywords': ['あーし', '今日', 'だし', 'じゃん']
        }
    ]
    
    # テスト実行
    logger.info("=" * 60)
    logger.info("テスト開始")
    logger.info("=" * 60)
    
    results = run_test(
        model,
        tokenizer,
        vectors,
        test_cases,
        strength=args.strength,
        max_tokens=args.max_tokens
    )
    
    # 結果表示
    logger.info("=" * 60)
    logger.info("テスト結果")
    logger.info("=" * 60)
    
    for result in results:
        print(f"\n【{result['name']}】")
        print(f"入力: {result['input']}")
        print(f"応答: {result['output']}")
        
        # キーワードチェック
        found_keywords = []
        for keyword in result['expected_keywords']:
            if keyword in result['output']:
                found_keywords.append(keyword)
        
        print(f"検出キーワード: {', '.join(found_keywords) if found_keywords else 'なし'}")
        print(f"キーワード含有率: {len(found_keywords)}/{len(result['expected_keywords'])}")
    
    return results


if __name__ == "__main__":
    results = main()
