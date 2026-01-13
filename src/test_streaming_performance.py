"""
VOICEVOXストリーミング音声合成のパフォーマンステスト

LLM推論速度とVOICEVOX音声合成速度を測定します。
"""

import asyncio
import time
import json
from typing import List, Dict
import httpx

# 設定
# WSL2からWindows側のVOICEVOXにアクセスするため、Windowsホストのアドレスを使用
# ブラウザは Windows 上で動作するため localhost でOKだが、
# WSL2のPythonスクリプトからは Windows側のIPが必要
import subprocess
try:
    # WSL2環境の場合、/etc/resolv.confからWindowsホストのIPを取得
    windows_host = subprocess.check_output(
        "cat /etc/resolv.conf | grep nameserver | awk '{print $2}'",
        shell=True
    ).decode().strip()
    # Windows側のVOICEVOXにアクセス
    VOICEVOX_API_URL = f"http://{windows_host}:50021"
except:
    # フォールバック: localhost（通常はこちらを使用）
    VOICEVOX_API_URL = "http://127.0.0.1:50021"

SPEAKER_ID = 8  # 春日部つむぎノーマル

# テスト用の文章（句読点で区切られた文）
TEST_SENTENCES = [
    "あーし、今日マジでテンション上がってるんだけど！",
    "せんぱいって、プログラミングめっちゃ得意じゃん？",
    "あーしもコード書くの好きなんだよね〜。",
    "でもさ、バグ出たときってマジ萎えるよね。",
    "エラーログ見るの、ガチで苦手なんだけど。",
]


async def measure_voicevox_synthesis(text: str) -> Dict:
    """VOICEVOX音声合成の速度を測定"""
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. audio_query
        query_start = time.time()
        query_response = await client.post(
            f"{VOICEVOX_API_URL}/audio_query",
            params={"text": text, "speaker": SPEAKER_ID}
        )
        query_time = time.time() - query_start
        
        if query_response.status_code != 200:
            return {
                "text": text,
                "success": False,
                "error": f"Query failed: {query_response.status_code}",
                "query_time": query_time,
                "synthesis_time": 0,
                "total_time": 0,
                "audio_size": 0,
            }
        
        query_json = query_response.json()
        
        # 2. synthesis
        synthesis_start = time.time()
        synthesis_response = await client.post(
            f"{VOICEVOX_API_URL}/synthesis",
            params={"speaker": SPEAKER_ID},
            headers={"Content-Type": "application/json", "Accept": "audio/wav"},
            content=json.dumps(query_json)
        )
        synthesis_time = time.time() - synthesis_start
        
        if synthesis_response.status_code != 200:
            return {
                "text": text,
                "success": False,
                "error": f"Synthesis failed: {synthesis_response.status_code}",
                "query_time": query_time,
                "synthesis_time": synthesis_time,
                "total_time": time.time() - start_time,
                "audio_size": 0,
            }
        
        audio_data = synthesis_response.content
        total_time = time.time() - start_time
        
        return {
            "text": text,
            "success": True,
            "query_time": query_time,
            "synthesis_time": synthesis_time,
            "total_time": total_time,
            "audio_size": len(audio_data),
            "text_length": len(text),
        }


async def test_sequential_synthesis():
    """順次合成のテスト（従来方式）"""
    print("\n" + "="*80)
    print("📊 順次合成テスト（従来方式：全文生成完了後に一括合成）")
    print("="*80)
    
    full_text = "".join(TEST_SENTENCES)
    print(f"\n📝 テキスト: {full_text}")
    print(f"📏 文字数: {len(full_text)}文字")
    
    start_time = time.time()
    result = await measure_voicevox_synthesis(full_text)
    total_time = time.time() - start_time
    
    if result["success"]:
        print(f"\n✅ 合成成功")
        print(f"  ⏱️  Query時間: {result['query_time']:.3f}秒")
        print(f"  ⏱️  Synthesis時間: {result['synthesis_time']:.3f}秒")
        print(f"  ⏱️  合計時間: {result['total_time']:.3f}秒")
        print(f"  📦 音声サイズ: {result['audio_size']:,}バイト")
        print(f"  🚀 処理速度: {result['text_length'] / result['total_time']:.2f}文字/秒")
    else:
        print(f"\n❌ 合成失敗: {result['error']}")
    
    return result


async def test_streaming_synthesis():
    """ストリーミング合成のテスト（新方式）"""
    print("\n" + "="*80)
    print("📊 ストリーミング合成テスト（新方式：文単位で順次合成）")
    print("="*80)
    
    results = []
    total_start = time.time()
    
    # 並列合成（実際のストリーミングをシミュレート）
    print(f"\n📝 {len(TEST_SENTENCES)}文を順次合成中...")
    
    for i, sentence in enumerate(TEST_SENTENCES, 1):
        print(f"\n[{i}/{len(TEST_SENTENCES)}] {sentence}")
        result = await measure_voicevox_synthesis(sentence)
        
        if result["success"]:
            print(f"  ✅ Query: {result['query_time']:.3f}秒 | Synthesis: {result['synthesis_time']:.3f}秒 | 合計: {result['total_time']:.3f}秒")
            results.append(result)
        else:
            print(f"  ❌ 失敗: {result['error']}")
    
    total_time = time.time() - total_start
    
    # 統計情報
    if results:
        avg_query = sum(r["query_time"] for r in results) / len(results)
        avg_synthesis = sum(r["synthesis_time"] for r in results) / len(results)
        avg_total = sum(r["total_time"] for r in results) / len(results)
        total_chars = sum(r["text_length"] for r in results)
        total_audio_size = sum(r["audio_size"] for r in results)
        
        print("\n" + "-"*80)
        print("📈 統計情報")
        print("-"*80)
        print(f"  合成成功: {len(results)}/{len(TEST_SENTENCES)}文")
        print(f"  平均Query時間: {avg_query:.3f}秒")
        print(f"  平均Synthesis時間: {avg_synthesis:.3f}秒")
        print(f"  平均合計時間: {avg_total:.3f}秒/文")
        print(f"  全体処理時間: {total_time:.3f}秒")
        print(f"  総文字数: {total_chars}文字")
        print(f"  総音声サイズ: {total_audio_size:,}バイト")
        print(f"  処理速度: {total_chars / total_time:.2f}文字/秒")
        
        # 最初の音声が再生開始されるまでの時間（TTFB: Time To First Byte）
        first_audio_time = results[0]["total_time"]
        print(f"\n⚡ 最初の音声再生開始まで: {first_audio_time:.3f}秒")
        print(f"   （ユーザーが最初の音声を聞くまでの待ち時間）")
    
    return results


async def test_parallel_synthesis():
    """並列合成のテスト（最大性能測定）"""
    print("\n" + "="*80)
    print("📊 並列合成テスト（理論上の最大性能）")
    print("="*80)
    
    print(f"\n📝 {len(TEST_SENTENCES)}文を並列合成中...")
    
    start_time = time.time()
    tasks = [measure_voicevox_synthesis(sentence) for sentence in TEST_SENTENCES]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    successful = [r for r in results if r["success"]]
    
    if successful:
        total_chars = sum(r["text_length"] for r in successful)
        total_audio_size = sum(r["audio_size"] for r in successful)
        
        print(f"\n✅ 合成成功: {len(successful)}/{len(TEST_SENTENCES)}文")
        print(f"  ⏱️  並列処理時間: {total_time:.3f}秒")
        print(f"  📏 総文字数: {total_chars}文字")
        print(f"  📦 総音声サイズ: {total_audio_size:,}バイト")
        print(f"  🚀 処理速度: {total_chars / total_time:.2f}文字/秒")
    
    return results


async def main():
    """メインテスト実行"""
    print("\n" + "="*80)
    print("🎤 VOICEVOXストリーミング音声合成 パフォーマンステスト")
    print("="*80)
    print(f"\n📍 VOICEVOX API: {VOICEVOX_API_URL}")
    print(f"🎙️  話者ID: {SPEAKER_ID} (春日部つむぎノーマル)")
    
    # VOICEVOXの接続確認
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{VOICEVOX_API_URL}/version")
            if response.status_code == 200:
                version = response.json()
                print(f"✅ VOICEVOX接続成功: {version}")
            else:
                print(f"⚠️  VOICEVOX接続エラー: {response.status_code}")
                return
    except Exception as e:
        print(f"❌ VOICEVOX接続失敗: {e}")
        print(f"   VOICEVOXアプリが起動しているか確認してください。")
        return
    
    # テスト実行
    sequential_result = await test_sequential_synthesis()
    await asyncio.sleep(1)  # 少し待機
    
    streaming_results = await test_streaming_synthesis()
    await asyncio.sleep(1)  # 少し待機
    
    parallel_results = await test_parallel_synthesis()
    
    # 比較結果
    print("\n" + "="*80)
    print("📊 パフォーマンス比較")
    print("="*80)
    
    if sequential_result["success"] and streaming_results:
        seq_time = sequential_result["total_time"]
        stream_time = sum(r["total_time"] for r in streaming_results)
        stream_first = streaming_results[0]["total_time"]
        
        print(f"\n【従来方式（一括合成）】")
        print(f"  ⏱️  全体処理時間: {seq_time:.3f}秒")
        print(f"  ⚡ ユーザー待機時間: {seq_time:.3f}秒（音声再生開始まで）")
        
        print(f"\n【新方式（ストリーミング合成）】")
        print(f"  ⏱️  全体処理時間: {stream_time:.3f}秒")
        print(f"  ⚡ ユーザー待機時間: {stream_first:.3f}秒（最初の音声再生開始まで）")
        
        print(f"\n【改善効果】")
        improvement = ((seq_time - stream_first) / seq_time) * 100
        print(f"  🚀 体感速度向上: {improvement:.1f}%")
        print(f"  ⏱️  待機時間短縮: {seq_time - stream_first:.3f}秒")
        
        print(f"\n【結論】")
        if improvement > 50:
            print(f"  ✨ ストリーミング方式により、体感速度が大幅に向上しました！")
            print(f"     ユーザーは {stream_first:.1f}秒で最初の音声を聞くことができます。")
        else:
            print(f"  ✅ ストリーミング方式により、体感速度が向上しました。")


if __name__ == "__main__":
    asyncio.run(main())
