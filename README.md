# Deep Learning ハンズオン教材

**フレームワークを使わず**、Python / NumPy だけでニューラルネットワークをゼロから実装し、
Google Colab 上で実際に動かして学習・可視化を体験する Deep Learning チュートリアルです。
上の「Open in Colab」バッジをクリックするだけで、環境構築なしにブラウザで動かせます。

## ノートブック一覧

| ノートブック | 内容 | データ | Colab |
|---|---|---|---|
| **DL01_xor.ipynb** | 純Python（NumPyなし）でXORを学習する2層NN。順伝播〜逆伝播の基礎 | 不要 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilu-h-bando/dl-tutorial/blob/main/DL01_xor.ipynb) |
| **DL30_optdigits_visual.ipynb** | NumPyで7層MLP。手書き数字(optdigits)分類と勾配・活性の可視化 | `optdigits/` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilu-h-bando/dl-tutorial/blob/main/DL30_optdigits_visual.ipynb) |

> リポジトリ名やオーナー名が上記（`ilu-h-bando/dl-tutorial`, ブランチ `main`）と違う場合は、
> README のバッジURLと、各ノートブック内の `GH_OWNER` / `GH_REPO` / `GH_BRANCH` を書き換えてください。

## 使い方

### 1. Colab で開く
上表の「Open in Colab」バッジをクリックします（初回はブラウザで Google にログイン）。
このリポジトリは公開なので、GitHub の認証やトークンは不要です。

### 2. 実行する
メニューの「ランタイム > すべてのセルを実行」を押すだけ。
`DL30` はデータ（`optdigits/`）を GitHub から自動ダウンロードしてから学習します。

### 3. いじってみる
セルの中の解説（Markdown）を読みながら、各ノート末尾の「やってみよう」で
パラメータ（学習率・層の深さ・活性化関数・ドロップアウトなど）を変えて、
挙動や可視化の違いを体感してください。

## リポジトリ構成

```
.
├─ DL01_xor.ipynb                # XOR（NumPyなし）
├─ DL30_optdigits_visual.ipynb   # optdigits 7層MLP + 可視化
├─ optdigits/                    # 訓練/検証データ（約0.8MB, そのままコミット）
│   ├─ optdigits.tra             # 訓練 3823件
│   ├─ optdigits.tes             # 検証 1797件
│   └─ optdigits.names           # データ説明
├─ requirements.txt              # numpy, matplotlib（Colabは導入済み）
└─ README.md
```

## メモ

- このリポジトリは **Public（公開）** です。Colab のバッジから誰でも認証なしに実行できます。
  社内限定に戻したい場合は Private に変更できますが、その場合は各利用者に Colab の
  GitHub 連携（プライベートリポジトリ許可）が必要になり、`optdigits/` の取得にも PAT が要ります。
- データ（`optdigits/`）は約0.8MBと小さいのでリポジトリに直接コミットしています。
  数十MBを超える場合は **Git LFS**、数GB規模なら Google Drive など外部ストレージを検討してください。
- フォークして自分のリポジトリで使う場合は、README のバッジURLと各ノートブック内の
  `GH_OWNER` / `GH_REPO` / `GH_BRANCH` を自分のものに書き換えてください。

## データの出典

`optdigits/` は UCI Machine Learning Repository の
[Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) データセットです。
