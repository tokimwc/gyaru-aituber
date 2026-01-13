"""
AITuberバックエンドのクイックテスト

修正版コードが正常に動作するか確認
"""

import torch
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 修正版のHook関数をインポート
import sys
sys.path.insert(0, 'src')
from run_gyaru_aituber import apply_gyaru_pre_hook, visualize_hook


def main():
    # 設定
    model_name = "models/Qwen2.5-32B-Instruct-bnb-4bit"
    vector_path = Path("outputs/vectors/Qwen2.5-32B-Instruct-bnb-4bit_gyaru_vector_manual.pt")
    strength = 15.0
    visualize_layer = 48
    
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
    
    # フック登録
    logger.info(f"フック登録中（strength={strength}）...")
    handles = []
    
    for layer_idx, vec in vectors.items():
        layer_module = model.model.layers[layer_idx]
        
        # Pre-Hook: ドーピング
        h1 = layer_module.register_forward_pre_hook(
            apply_gyaru_pre_hook(vec, strength)
        )
        handles.append(h1)
        
        # Post-Hook: 可視化（指定レイヤーのみ）
        if layer_idx == visualize_layer:
            h2 = layer_module.register_forward_hook(
                visualize_hook(vec)
            )
            handles.append(h2)
    
    logger.info(f"✅ フック登録完了: {len(handles)}個")
    
    # テストケース
    test_input = "こんにちは"
    
    messages = [
        {
            "role": "system",
            "content": "あなたは快活なAITuber「りんね」です。一人称は必ず「あーし」を使ってください。技術的な話も得意ですが、口調は常にギャル語です。"
        },
        {"role": "user", "content": test_input}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    logger.info("=" * 60)
    logger.info("生成テスト開始")
    logger.info("=" * 60)
    logger.info(f"入力: {test_input}")
    logger.info("生成中...（可視化バーが表示されるはず）")
    print()  # 可視化バーのための空行
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    print()  # 可視化バーの後に改行
    logger.info("=" * 60)
    logger.info("生成結果")
    logger.info("=" * 60)
    logger.info(f"応答: {generated_text}")
    
    # キーワードチェック
    gyaru_keywords = ['あーし', 'マジ', 'ガチ', 'だし', 'じゃん', 'エグ', '草']
    found = [kw for kw in gyaru_keywords if kw in generated_text]
    logger.info(f"ギャルキーワード: {found if found else 'なし'}")
    
    # フック解除
    for h in handles:
        h.remove()
    
    logger.info("✅ テスト完了")


if __name__ == "__main__":
    main()
