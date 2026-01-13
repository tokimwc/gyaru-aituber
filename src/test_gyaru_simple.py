"""
シンプルなギャルAITuberテスト

Hookの戻り値を返さない方法で、直接的なテストを行う
"""

import torch
import logging
from pathlib import Path
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    logger.info("ギャルベクトルなしでテスト")
    logger.info("=" * 60)
    
    # まずはベクトルなしで生成
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
        
        logger.info(f"応答（ノーマル）: {generated_text}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ ベクトルなしテスト完了")


if __name__ == "__main__":
    main()
