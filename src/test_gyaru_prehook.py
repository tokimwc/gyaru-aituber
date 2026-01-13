"""
Pre-Hookを使ったギャルベクトル注入テスト

forward_hookではなくpre_hookを使って、レイヤーの入力側で介入
"""

import torch
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def apply_gyaru_pre_hook(layer_idx: int, vector: torch.Tensor, strength: float):
    """
    Pre-Hook: レイヤーの入力に介入
    """
    def hook(module, input):
        # inputはタプル: (hidden_states,) または (hidden_states, attention_mask, ...)
        if isinstance(input, tuple) and len(input) > 0:
            hidden = input[0]
            if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
                # [batch, seq, hidden_dim]
                intervention = vector * strength
                # 新しいhidden_statesを作成
                new_hidden = hidden + intervention
                # 新しいinputタプルを返す
                return (new_hidden,) + input[1:]
        return input
    return hook


def test_with_prehook(model, tokenizer, vectors, test_inputs, strength=1.5):
    """Pre-Hookを使ったテスト"""
    device = next(model.parameters()).device
    
    # Pre-Hookを登録
    handles = []
    for layer_idx, vector in vectors.items():
        layer = model.model.layers[layer_idx]
        h = layer.register_forward_pre_hook(
            apply_gyaru_pre_hook(layer_idx, vector.to(device), strength)
        )
        handles.append(h)
    
    logger.info(f"✅ Pre-Hook登録完了: {len(handles)}レイヤー")
    
    results = []
    
    try:
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
            
            logger.info(f"応答（ギャル）: {generated_text}")
            results.append({
                'input': user_input,
                'output': generated_text
            })
    
    finally:
        # Hook解除
        for h in handles:
            h.remove()
        logger.info("✅ Pre-Hook解除完了")
    
    return results


def main():
    # 設定
    model_name = "models/Qwen2.5-32B-Instruct-bnb-4bit"
    vector_path = Path("outputs/vectors/Qwen2.5-32B-Instruct-bnb-4bit_gyaru_vector_manual.pt")
    
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
    
    # テストケース
    test_inputs = [
        "こんにちは",
        "Pythonでエラーが出るんだけど",
        "今日何してた？"
    ]
    
    logger.info("=" * 60)
    logger.info("Pre-Hookでギャルベクトル注入テスト")
    logger.info("=" * 60)
    
    # strength=1.5でテスト
    logger.info("\n【strength=1.5】")
    results_15 = test_with_prehook(model, tokenizer, vectors, test_inputs, strength=1.5)
    
    # strength=3.0でテスト
    logger.info("\n" + "=" * 60)
    logger.info("【strength=3.0】（強め）")
    logger.info("=" * 60)
    results_30 = test_with_prehook(model, tokenizer, vectors, test_inputs, strength=3.0)
    
    # 結果比較
    logger.info("\n" + "=" * 60)
    logger.info("結果比較")
    logger.info("=" * 60)
    
    for i, (r15, r30) in enumerate(zip(results_15, results_30), 1):
        print(f"\n【テスト{i}】: {r15['input']}")
        print(f"strength=1.5: {r15['output'][:80]}...")
        print(f"strength=3.0: {r30['output'][:80]}...")
        
        # キーワードチェック
        gyaru_keywords = ['あーし', 'マジ', 'ガチ', 'だし', 'じゃん', 'エグ', '草', 'うぃーす', 'おっはー']
        found_15 = [kw for kw in gyaru_keywords if kw in r15['output']]
        found_30 = [kw for kw in gyaru_keywords if kw in r30['output']]
        print(f"ギャルキーワード(1.5): {', '.join(found_15) if found_15 else 'なし'}")
        print(f"ギャルキーワード(3.0): {', '.join(found_30) if found_30 else 'なし'}")
    
    logger.info("\n✅ テスト完了")


if __name__ == "__main__":
    main()
