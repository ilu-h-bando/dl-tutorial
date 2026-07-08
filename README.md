# Deep Learning ハンズオン教材

社内技術者向けの Deep Learning チュートリアルです。**フレームワークを使わず**、Python / NumPy だけで
ニューラルネットワークをゼロから実装し、Google Colab 上で実際に動かして学習・可視化を体験します。

## ノートブック一覧

| ノートブック | 内容 | データ | Colab |
|---|---|---|---|
| **DL01_xor.ipynb** | 純Python（NumPyなし）でXORを学習する2層NN。順伝播〜逆伝播の基礎 | 不要 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilu-h-bando/dl-tutorial/blob/main/DL01_xor.ipynb) |
| **DL30_optdigits_visual.ipynb** | NumPyで7層MLP。手書き数字(optdigits)分類と勾配・活性の可視化 | `optdigits/` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilu-h-bando/dl-tutorial/blob/main/DL30_optdigits_visual.ipynb) |

> リポジトリ名やオーナー名が上記（`ilu-h-bando/dl-tutorial`, ブランチ `main`）と違う場合は、
> README のバッジURLと、各ノートブック内の `GH_OWNER` / `GH_REPO` / `GH_BRANCH` を書き換えてください。

## 使い方（受講者向け）

### 1. アクセス権をもらう
このリポジトリはプライベートです。管理者から GitHub への招待（メール）を受け取り、
自分の GitHub アカウントで承認してください。

### 2. Colab で開く
上表の「Open in Colab」バッジをクリックします。初回はブラウザで Google にログインし、
Colab から GitHub への連携（**プライベートリポジトリを含む**アクセス許可）を承認してください。

### 3. （DL30のみ）データ取得のための Token 登録
`DL30_optdigits_visual.ipynb` はプライベートリポジトリ内の `optdigits/` を読むため、
**GitHub Personal Access Token (PAT)** が必要です。

1. GitHub → Settings → Developer settings → Personal access tokens で PAT を発行（`repo` 読み取り権限で十分）
2. Colab 画面左の **🔑（シークレット）** をクリック
3. 「新しいシークレットを追加」→ **名前: `GH_TOKEN`**、値に PAT を貼り付け
4. その項目の「ノートブックからのアクセス」を **ON**

> `DL01_xor.ipynb` は外部データ不要なので、この手順は不要です。

### 4. 実行する
メニューの「ランタイム > すべてのセルを実行」。セルの中の解説（Markdown）を読みながら、
末尾の「やってみよう」でパラメータを変えて挙動の違いを体感してください。

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

## 管理者向けメモ（アクセス制限の運用）

- リポジトリを **Private** にし、受講者を **メールアドレスで招待**（各自のGitHubアカウントで承認）。
  人数が多い場合は **Organization + Team** での一括管理を推奨。
- データ（`optdigits/`）は約0.8MBと小さいのでリポジトリに直接コミット。
  数十MBを超える場合は **Git LFS**、数GB規模なら Google Drive など外部ストレージを検討。
- Colab はプライベートリポジトリのノートブックも開けます（利用者がGitHub連携を許可すれば可）。
  ノートブック内からのデータ取得は、各自の PAT を Colab シークレット `GH_TOKEN` に登録する方式です。
