"""
Gyaru Dataset Generator - Main Script (vLLM API版)

vLLM OpenAI互換APIを使用して「あーし系ギャル」データセットを生成
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import yaml
from openai import OpenAI
from rich.console import Console
from rich.progress import track
from pydantic import BaseModel

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()


class DatasetConfig(BaseModel):
    """設定クラス"""
    config_path: Path = Path("config/generation_config.yaml")
    api_base: str = "http://localhost:8000/v1"
    num_pairs_per_batch: int = 30
    total_batches: int = 4
    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 2000
    output_dir: Path = Path("outputs")
    require_ash: bool = True


class GyaruDatasetGenerator:
    """ギャルデータセット生成器（vLLM API版）"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.console = Console()
        
        # 設定ファイル読み込み
        self.model_name = None
        if config.config_path.exists():
            with open(config.config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                # モデルパスを読み込み
                if 'model' in yaml_config and 'path' in yaml_config['model']:
                    self.model_name = yaml_config['model']['path']
                # YAML設定から値を上書き
                if 'vllm' in yaml_config:
                    self.config.api_base = yaml_config['vllm'].get('api_base', self.config.api_base)
                if 'generation' in yaml_config:
                    gen_config = yaml_config['generation']
                    self.config.num_pairs_per_batch = gen_config.get('num_pairs_per_batch', self.config.num_pairs_per_batch)
                    self.config.total_batches = gen_config.get('total_batches', self.config.total_batches)
                    self.config.temperature = gen_config.get('temperature', self.config.temperature)
                    self.config.top_p = gen_config.get('top_p', self.config.top_p)
                    self.config.max_new_tokens = gen_config.get('max_new_tokens', self.config.max_new_tokens)
                if 'output' in yaml_config:
                    self.config.output_dir = Path(yaml_config['output'].get('output_dir', self.config.output_dir))
                if 'validation' in yaml_config:
                    self.config.require_ash = yaml_config['validation'].get('require_ash', self.config.require_ash)
        
        # モデル名が設定されていない場合のデフォルト値
        if not self.model_name:
            self.model_name = "models/Qwen2.5-32B-Instruct-bnb-4bit"
        
        # プロンプト読み込み
        prompt_path = Path("prompts/system_prompt.txt")
        if not prompt_path.exists():
            raise FileNotFoundError(f"プロンプトファイルが見つかりません: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()
        
        # OpenAI互換APIクライアント初期化
        logger.info(f"🔌 Connecting to vLLM API: {self.config.api_base}")
        self.client = OpenAI(
            base_url=self.config.api_base,
            api_key="not-needed"  # vLLMはAPIキー不要
        )
        
        # サーバー接続確認
        self._check_server_connection()
    
    def _check_server_connection(self, max_retries: int = 5, retry_delay: int = 2):
        """vLLMサーバーへの接続を確認"""
        for i in range(max_retries):
            try:
                # 簡単なリクエストで接続確認
                models = self.client.models.list()
                logger.info("✅ vLLMサーバーに接続成功")
                return True
            except Exception as e:
                if i < max_retries - 1:
                    logger.warning(f"⚠️  サーバー接続失敗 (試行 {i+1}/{max_retries}): {e}")
                    logger.info(f"⏳ {retry_delay}秒後に再試行...")
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(
                        f"❌ vLLMサーバーに接続できません: {self.config.api_base}\n"
                        f"   サーバーが起動しているか確認してください: python src/start_vllm_server.py"
                    ) from e
        return False
    
    def generate_batch(self, batch_num: int) -> List[Dict]:
        """1バッチ分のデータ生成"""
        logger.info(f"バッチ {batch_num}/{self.config.total_batches} を生成中...")
        
        # メッセージ形式でプロンプト構築
        messages = [
            {"role": "system", "content": "あなたは優秀なシナリオライターです。JSON形式でのみ応答してください。"},
            {"role": "user", "content": self.system_prompt}
        ]
        
        # vLLM APIで生成
        try:
            # モデル名は設定ファイルから読み込んだ値を使用
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_new_tokens
            )
            
            generated_text = response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ API呼び出しエラー (バッチ {batch_num}): {e}")
            return []
        
        # JSON抽出
        try:
            # ```json ... ``` で囲まれている場合を処理
            if "```json" in generated_text:
                json_start = generated_text.find("```json") + 7
                json_end = generated_text.find("```", json_start)
                json_str = generated_text[json_start:json_end].strip()
            elif "```" in generated_text:
                # ``` で囲まれている場合（言語指定なし）
                json_start = generated_text.find("```") + 3
                json_end = generated_text.find("```", json_start)
                json_str = generated_text[json_start:json_end].strip()
            elif "[" in generated_text:
                # [ から始まる部分を抽出
                json_start = generated_text.find("[")
                json_end = generated_text.rfind("]") + 1
                json_str = generated_text[json_start:json_end]
            else:
                json_str = generated_text.strip()
            
            data = json.loads(json_str)
            
            # リストでない場合はリストに変換
            if not isinstance(data, list):
                data = [data]
            
            # IDを追加
            for i, item in enumerate(data):
                item["id"] = (batch_num - 1) * self.config.num_pairs_per_batch + i + 1
                item["batch"] = batch_num
                item["generated_at"] = datetime.now().isoformat()
            
            logger.info(f"✅ バッチ {batch_num}: {len(data)}ペア生成成功")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON解析エラー (バッチ {batch_num}): {e}")
            logger.error(f"生成されたテキスト:\n{generated_text[:500]}...")
            return []
    
    def validate_dataset(self, data: List[Dict]) -> List[Dict]:
        """データセットの品質検証"""
        validated = []
        for item in data:
            # 必須フィールドチェック
            if not all(k in item for k in ["topic", "standard", "gyaru"]):
                logger.warning(f"❌ 無効なデータ (ID: {item.get('id', 'N/A')}): 必須フィールド不足")
                continue
            
            # 「あーし」チェック
            if self.config.require_ash and "あーし" not in item["gyaru"]:
                logger.warning(f"⚠️  「あーし」なし (ID: {item['id']}): {item['gyaru'][:50]}...")
                # 警告だけ出して、データは残す
            
            validated.append(item)
        
        logger.info(f"✅ 検証完了: {len(validated)}/{len(data)}ペアが有効")
        return validated
    
    def generate_dataset(self) -> Path:
        """データセット全体を生成"""
        all_data = []
        
        self.console.print(f"[bold cyan]🚀 データセット生成開始[/bold cyan]")
        self.console.print(f"目標: {self.config.total_batches}バッチ × {self.config.num_pairs_per_batch}ペア = {self.config.total_batches * self.config.num_pairs_per_batch}ペア\n")
        
        for batch_num in track(range(1, self.config.total_batches + 1), description="生成中..."):
            batch_data = self.generate_batch(batch_num)
            if batch_data:
                all_data.extend(batch_data)
                
                # バッチごとに保存 (バックアップ)
                batch_file = self.config.output_dir / "raw" / f"batch_{batch_num:02d}.json"
                batch_file.parent.mkdir(parents=True, exist_ok=True)
                with open(batch_file, 'w', encoding='utf-8') as f:
                    json.dump(batch_data, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 バッチ {batch_num} を保存: {batch_file}")
        
        # 検証
        validated_data = self.validate_dataset(all_data)
        
        # 最終出力
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.config.output_dir / "processed" / f"gyaru_dataset_{timestamp}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validated_data, f, ensure_ascii=False, indent=2)
        
        self.console.print(f"\n[bold green]🎉 データセット生成完了！[/bold green]")
        self.console.print(f"[green]📁 ファイル: {output_file}[/green]")
        self.console.print(f"[green]📊 総ペア数: {len(validated_data)}[/green]")
        
        # 統計情報
        topics = {}
        for item in validated_data:
            topic = item.get("topic", "Unknown")
            topics[topic] = topics.get(topic, 0) + 1
        
        self.console.print("\n[cyan]📈 トピック別統計:[/cyan]")
        for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True):
            self.console.print(f"  {topic}: {count}ペア")
        
        return output_file


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gyaru Dataset Generator (vLLM API版)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/generation_config.yaml"),
        help="設定ファイルのパス"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="vLLM APIベースURL (デフォルト: 設定ファイルから読み込み)"
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=None,
        help="生成バッチ数 (デフォルト: 設定ファイルから読み込み)"
    )
    
    args = parser.parse_args()
    
    config = DatasetConfig(config_path=args.config)
    if args.api_base:
        config.api_base = args.api_base
    if args.batches:
        config.total_batches = args.batches
    
    generator = GyaruDatasetGenerator(config)
    
    try:
        output_file = generator.generate_dataset()
        console.print(f"\n[bold green]✅ 成功！ {output_file} を確認してください[/bold green]")
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}", exc_info=True)
        console.print(f"[bold red]❌ エラー: {e}[/bold red]")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
