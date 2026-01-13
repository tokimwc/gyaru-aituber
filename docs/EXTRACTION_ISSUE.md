# ベクトル抽出の課題

## 問題

repengライブラリを使用してQwen2.5-32B-Instruct-bnb-4bitモデルからベクトルを抽出しようとした際、以下のエラーが発生しました:

```
AttributeError: 'ControlModule' object has no attribute 'attention_type'
```

## エラー詳細

- **発生箇所**: `repeng/control.py` の `ControlModel` がQwen2.5の内部構造にアクセスしようとした際
- **原因**: repengの `ControlModule` がQwen2.5の `attention_type` 属性にアクセスできない
- **影響**: ベクトル抽出が実行できない

## 試行した解決策

1. ✅ repengを最新版（0.5.0）にアップグレード
2. ✅ モデルのロード方法を変更（4bit量子化の再適用を回避）
3. ✅ batch_sizeを1に削減
4. ✅ データ数を50件に削減
5. ❌ いずれも同じエラーが発生

## 現在の状況

- repengがQwen2.5を完全にサポートしていない可能性
- または、Qwen2.5の特定のバージョン/量子化形式との互換性問題

## 解決済み

**解決方法**: 手動実装（PyTorch Hook + scikit-learn PCA）

**結果**:
- ✅ 50ペアのデータから31レイヤー分のベクトルを抽出成功
- ✅ 出力: `outputs/vectors/Qwen2.5-32B-Instruct-bnb-4bit_gyaru_vector_manual.pt` (321KB)
- ✅ 詳細は `docs/VECTOR_EXTRACTION.md` を参照

## 関連ファイル

- `src/extract_gyaru_vector.py` - repeng版（互換性問題で未使用）
- `src/extract_gyaru_vector_manual.py` - 手動実装版（成功）
- `outputs/vectors/Qwen2.5-32B-Instruct-bnb-4bit_gyaru_vector_manual.pt` - 抽出されたベクトル
- `outputs/vectors/Qwen2.5-32B-Instruct-bnb-4bit_gyaru_vector_stats.txt` - 統計情報

## 参考情報

- repeng GitHub: https://github.com/vgel/repeng
- 使用モデル: Qwen2.5-32B-Instruct-bnb-4bit
- repengバージョン: 0.5.0 (GitHubから直接インストール)
- 解決方法: PyTorch Hook + scikit-learn PCA