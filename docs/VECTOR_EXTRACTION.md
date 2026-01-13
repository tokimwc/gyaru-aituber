# ベクトル抽出

## 概要

Representation Engineering (RepE) を使用して、Qwen2.5-32B-Instruct-bnb-4bitモデルから「ギャル成分」を表す制御ベクトルを抽出しました。

## 実装方法

### 試行1: repengライブラリ（失敗）

**使用ライブラリ**: repeng 0.5.0

**問題**: Qwen2.5との互換性問題
- `AttributeError: 'ControlModule' object has no attribute 'attention_type'`
- repengの`ControlModule`がQwen2.5の新しい実装に対応していない

**試行した対策**:
1. repengを最新版（0.5.0）にアップグレード → 失敗
2. モンキーパッチで`__getattr__`を追加 → 失敗

### 試行2: 手動実装（成功）

**使用技術**:
- PyTorch Hook (`register_forward_hook()`)
- scikit-learn PCA

**実装手順**:
1. データセット読み込み（117ペア → 50ペアに制限）
2. 各ペアに対して:
   - Standard（標準語）とGyaru（ギャル口調）の入力を準備
   - Chat Templateを適用
   - 各レイヤーにHookを登録
   - 推論を実行してHidden Statesを取得
   - 差分（Gyaru - Standard）を計算
3. 各レイヤーごとにPCAで第1主成分を抽出
4. 符号を調整（平均との内積が正になるように）
5. PyTorch形式で保存

## 抽出結果

### 基本情報

| 項目 | 値 |
|------|-----|
| **使用モデル** | Qwen2.5-32B-Instruct-bnb-4bit |
| **データ数** | 50ペア |
| **対象レイヤー** | 20-50（31レイヤー） |
| **ベクトル次元** | 5120次元 |
| **出力ファイル** | `Qwen2.5-32B-Instruct-bnb-4bit_gyaru_vector_manual.pt` (321KB) |

### レイヤー別分散説明率

各レイヤーの第1主成分が説明する分散の割合:

| レイヤー範囲 | 分散説明率 | 特徴 |
|-------------|-----------|------|
| 20-23 | 12.6-14.2% | 前半、やや高い |
| 24-30 | 10.4-12.1% | 中盤、低下傾向 |
| 31-40 | 10.1-11.3% | 中盤、安定 |
| 41-50 | 10.8-16.4% | 後半、上昇傾向 |

**最高**: レイヤー49（16.36%）  
**最低**: レイヤー39（10.09%）

**傾向**: 後半のレイヤーほど分散説明率が高い。これは、スタイル情報が深いレイヤーで表現されることを示唆。

### ベクトル統計

全レイヤーで一貫した統計:
- **Norm**: 1.0000（正規化済み）
- **Mean**: -0.0003 〜 0.0002（ほぼゼロ）
- **Std**: 0.0140（一定）

## 使用方法

### ベクトルの読み込み

```python
import torch

# ベクトルの読み込み
vectors = torch.load("outputs/vectors/Qwen2.5-32B-Instruct-bnb-4bit_gyaru_vector_manual.pt")

# 特定のレイヤーのベクトルを取得
layer_30_vector = vectors[30]  # shape: [5120]
```

### モデルへの適用（概念）

```python
# 推論時にHidden Statesに加算
# 例: レイヤー30に強度0.5で適用
strength = 0.5
gyaru_vector = vectors[30] * strength

# Hookを使用して推論時に加算
def add_vector_hook(module, input, output):
    if isinstance(output, tuple):
        hidden = output[0]
    else:
        hidden = output
    # ベクトルを加算
    hidden[:, -1, :] += gyaru_vector.to(hidden.device)
    return output

# レイヤーにHookを登録
handle = model.model.layers[30].register_forward_hook(add_vector_hook)

# 推論実行
output = model.generate(...)

# Hook解除
handle.remove()
```

## 検証

### データ品質

- **使用データ**: 117ペア中50ペアを使用
- **「あーし」含有率**: 約92%
- **トピック分布**: Tech, Streaming, Daily, Gaming が均等

### PCA品質

- **分散説明率**: 10-16%（妥当な範囲）
- **レイヤー間の一貫性**: 全レイヤーで正規化されたベクトルを抽出
- **符号の一貫性**: 平均との内積が正になるように調整済み

## 技術的詳細

### Hidden Statesの取得

```python
def get_hook(storage_list):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # 最後のトークンの隠れ状態を取得
        storage_list.append(hidden[:, -1, :].detach().cpu())
    return hook
```

### PCAによる主成分抽出

```python
from sklearn.decomposition import PCA

# 差分データを集約
X = torch.cat(diff_list, dim=0).float().numpy()  # [num_samples, hidden_dim]

# PCA
pca = PCA(n_components=1)
pca.fit(X)

# 第1主成分
direction = pca.components_[0]  # [hidden_dim]

# 符号調整
mean_diff = np.mean(X, axis=0)
if np.dot(direction, mean_diff) < 0:
    direction = -direction
```

## パフォーマンス

| 項目 | 時間 |
|------|------|
| モデルロード | 約1分40秒 |
| Hidden States抽出 | 約18秒（50ペア） |
| PCA計算 | 約3秒 |
| **合計** | **約2分** |

**VRAM使用量**: 約29GB / 32GB

## 今後の展開

1. **ベクトルの検証**: 実際にモデルに適用して生成テスト
2. **最適なレイヤーの特定**: 各レイヤーの効果を比較
3. **強度の調整**: ベクトルの強度を変えて効果を測定
4. **複数レイヤーの組み合わせ**: 複数のレイヤーに同時適用
5. **データ数の増加**: 全117ペアを使用して再抽出

## 参考文献

- Representation Engineering: https://github.com/vgel/repeng
- PyTorch Hooks: https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html
- PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
