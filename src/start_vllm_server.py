"""
vLLM Server Startup Script for Gyaru Dataset Generator

GPTQ-Int8モデルをvLLMで起動するスクリプト
vLLMの標準OpenAI互換APIサーバーを使用
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path
import yaml

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path = Path("config/generation_config.yaml")):
    """設定ファイルを読み込む"""
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def start_vllm_server(config_path: Path = None):
    """vLLMサーバーを起動（vLLMの標準APIサーバーを使用）"""
    if config_path is None:
        config_path = Path("config/generation_config.yaml")
    
    config = load_config(config_path)
    model_config = config['model']
    vllm_config = config['vllm']
    
    logger.info("=" * 60)
    logger.info("🚀 vLLM Server Starting...")
    logger.info("=" * 60)
    logger.info(f"📦 Model: {model_config['path']}")
    logger.info(f"🔧 Quantization: {vllm_config['quantization']}")
    logger.info(f"🌐 Host: {vllm_config['host']}:{vllm_config['port']}")
    logger.info("=" * 60)
    
    # vLLMの標準OpenAI互換APIサーバーを起動
    # uv run python -m vllm.entrypoints.openai.api_server を使用
    import sys
    import shutil
    
    # uv runコマンドを使用（vLLMがuv環境にインストールされている場合）
    uv_path = shutil.which("uv")
    if uv_path:
        cmd = [
            "uv", "run", "--with", "vllm", "--with", "bitsandbytes", "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_config['path'],
            "--host", vllm_config['host'],
            "--port", str(vllm_config['port']),
            "--tensor-parallel-size", str(vllm_config['tensor_parallel_size']),
            "--gpu-memory-utilization", str(vllm_config['gpu_memory_utilization']),
            "--max-model-len", str(vllm_config['max_model_len']),
            "--quantization", vllm_config['quantization'],
            "--dtype", "float16",  # GPTQはfloat16のみサポート
        ]
    else:
        # uvがない場合は通常のpythonを使用
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_config['path'],
            "--host", vllm_config['host'],
            "--port", str(vllm_config['port']),
            "--tensor-parallel-size", str(vllm_config['tensor_parallel_size']),
            "--gpu-memory-utilization", str(vllm_config['gpu_memory_utilization']),
            "--max-model-len", str(vllm_config['max_model_len']),
            "--quantization", vllm_config['quantization'],
            "--dtype", "float16",  # GPTQはfloat16のみサポート
        ]
    
    if vllm_config['trust_remote_code']:
        cmd.append("--trust-remote-code")
    
    logger.info(f"⏳ Starting vLLM server...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # サーバーを起動（ブロッキング）
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("\n👋 Server stopped by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="vLLM Server for Gyaru Dataset Generator")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/generation_config.yaml"),
        help="設定ファイルのパス"
    )
    
    args = parser.parse_args()
    
    try:
        start_vllm_server(args.config)
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
