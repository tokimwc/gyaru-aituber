"""
ギャルベクトル注入テスト

モデルの重みを直接変更する方法でベクトルを注入
"""

import torch
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def inject_gyaru_vectors(model, vectors, strength=1.5):
    """
    モデルの各レイヤーの出力に直接ベクトルを加算する
    （重みを一時的に変更する方法）
    """
    # 元の重みをバックアップ
    original_weights = {}
    
    for layer_idx, vector in vectors.items():
        layer = model.model.layers[layer_idx]
        
        # post_attention_layernormの重みを調整
        # これにより、レイヤーの出力が変化する
        if hasattr(layer, 'post_attention_layernorm'):
            # 元の重みを保存
            original_weights[layer_idx] = {
                'weight': layer.post_attention_layernorm.weight.data.clone(),
            }
            
            # ベクトルを重みに加算（簡易的な方法）
            # 注: これは実験的な実装
            bias_term = vector * strength * 0.01  # 小さな係数で調整
            layer.post_attention_layernorm.weight.data += bias_term
    
    return original_weights


def restore_weights(model, original_weights):
    """元の重みを復元"""
    for layer_idx, weights in original_weights.items():
        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'post_attention_layernorm'):
            layer.post_attention_layernorm.weight.data = weights['weight']


def test_generation(model, tokenizer, test_inputs, mode="normal"):
    """生成テスト"""
    device = next(model.parameters()).device
    results = []
    
    for i, user_input in enumerate(test_inputs, 1):
        logger.info(f"\nテスト {i}/{len(test_inputs)}: {user_input}")
        
        messages = [
            {"role": "system", "content": "あなたはAITuberのりんねです。雑談配信をしています。"},
            {"role": "user", "content": user_input}
        ]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        logger.info(f"応答（{mode}）: {generated_text}")
        results.append({
            'input': user_input,
            'output': generated_text,
            'mode': mode
        })
    
    return results


def main():
    # 設定
    model_name = "models/Qwen2.5-32B-Instruct-bnb-4bit"
    vector_path = Path("outputs/vectors/Qwen2.5-32B-Instruct-bnb-4bit_gyaru_vector_manual.pt")
    strength = 1.5
    
    # ベクトル読み込み
    logger.info(f"ベクトル読み込み: {vector_path}")
    vectors = torch.load(vector_path)
    logger.info(f"ベクトル読み込み完了: {len(vectors)}レイヤー")
    
    # モデル読み込み
    logger.info(f"モデル読み込み: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.eval()
    logger.info("モデル読み込み完了")
    
    device = next(model.parameters()).device
    
    # ベクトルをデバイスに転送
    for layer_idx in vectors:
        vectors[layer_idx] = vectors[layer_idx].to(device)
    
    # テストケース
    test_inputs = [
        "こんにちは",
        "Pythonでエラーが出るんだけど",
        "今日何してた？"
    ]
    
    logger.info("=" * 60)
    logger.info("1. ノーマルモードでテスト")
    logger.info("=" * 60)
    
    normal_results = test_generation(model, tokenizer, test_inputs, mode="ノーマル")
    
    logger.info("\n" + "=" * 60)
    logger.info("2. ギャルモードでテスト（重み変更方式）")
    logger.info("=" * 60)
    
    # ベクトル注入
    logger.info(f"ギャルベクトルを注入中（strength={strength}）...")
    original_weights = inject_gyaru_vectors(model, vectors, strength)
    logger.info("注入完了")
    
    gyaru_results = test_generation(model, tokenizer, test_inputs, mode="ギャル")
    
    # 重みを復元
    logger.info("\n元の重みを復元中...")
    restore_weights(model, original_weights)
    logger.info("復元完了")
    
    logger.info("\n" + "=" * 60)
    logger.info("結果比較")
    logger.info("=" * 60)
    
    for i, (normal, gyaru) in enumerate(zip(normal_results, gyaru_results), 1):
        print(f"\n【テスト{i}】: {normal['input']}")
        print(f"ノーマル: {normal['output'][:80]}...")
        print(f"ギャル  : {gyaru['output'][:80]}...")
        
        # キーワードチェック
        gyaru_keywords = ['あーし', 'マジ', 'ガチ', 'だし', 'じゃん', 'エグ', '草']
        found = [kw for kw in gyaru_keywords if kw in gyaru['output']]
        print(f"ギャルキーワード: {', '.join(found) if found else 'なし'}")
    
    logger.info("\n✅ テスト完了")


if __name__ == "__main__":
    main()
