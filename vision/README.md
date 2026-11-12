# Vision Transformer・Vision-Language Model ハンズオン教材

[transformer/](../transformer/) の[TF01](../transformer/TF01_tiny_gpt_waka.ipynb)（1ブロックのGPT）の部品を、**画像**に使うチュートリアルです。
TF02は小さなVision Transformer（ViT）で服・靴・バッグなどを分類します。
TF03はTF02に文字の生成を足した小さなVision-Language Model（VLM）で、画像を見て種類をカタカナで答えます。
PyTorchやTensorFlow、学習済みモデルは使いません。Google Colabのボタンからブラウザで実行できます。

## ノートブック

| ノートブック | 内容 | データ | Colab |
|---|---|---|---|
| **[TF02_vision_transformer_fashion.ipynb](TF02_vision_transformer_fashion.ipynb)** | NumPyで1ブロックのViTを実装。パッチ化・CLS・位置情報・Attentionを観察し、Fashion-MNISTを10分類する | 実行時にFashion-MNIST公式GitHubから自動取得 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilu-h-bando/dl-tutorial/blob/main/vision/TF02_vision_transformer_fashion.ipynb) |
| **[TF03_vision_language_model_fashion.ipynb](TF03_vision_language_model_fashion.ipynb)** | NumPyで1ブロックのVLMを実装。画像パッチと文字を1本の列に並べ、Fashion-MNISTの種類を「スニーカー」などのカタカナで1文字ずつ生成して答える | 実行時にFashion-MNIST公式GitHubから自動取得 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilu-h-bando/dl-tutorial/blob/main/vision/TF03_vision_language_model_fashion.ipynb) |

## TF02 Fashion-MNISTを分類するVision Transformer

TF01の64次元・4ヘッド・SwiGLU・逆伝播・AdamWを再利用し、和歌の次文字予測を画像の10分類に置き換えます。

```text
28×28画像 → パッチ分割 → Linearで64次元へ → CLSと位置Embedding
  → マスクなしSelf-Attention → SwiGLU MLP → CLSから10クラスを分類
```

初期設定は **PATCH=7、16パッチ＋CLSの17トークン、70,922パラメータ**です。
元来のViTのMLPはGELUですが、この教材はTF01との対応を追えるようSwiGLUを使った変種です。

### 実行する

1. 上表のTF02のColabボタンで開き、「ランタイム > すべてのセルを実行」を選びます。
2. 公式訓練60,000枚を、クラスごとに**学習54,000枚・検証6,000枚**へ分割します。
3. 各学習画像をエポックごとに1回ずつ使います。公式テスト10,000枚は学習やモデル選択には使いません。
4. 学習前後の精度・画像予測・混同行列・各ヘッドのAttentionを確認します。

NumPyのCPU実装です。ColabでGPUを選択しても、このコードはGPUを使いません。
初回はデータ取得のためインターネット接続が必要で、`fashion_mnist/` にキャッシュします。

### パッチサイズを変える

設定セルの `PATCH` は **1・2・4・7・14・28** から選べます。変更後は設定セルから全て再実行してください。
PATCHはパッチ数ではなく、**パッチの一辺の画素数**です。例えば14は14×14画素の4パッチ、28は画像全体の1パッチです。
小さいパッチではメモリ対策として学習・評価バッチ数を自動調整します。

初期設定は `EPOCHS=8`、`MAX_TRAIN_SECONDS=300.0` です。学習ループは8エポックまたは300秒で終了します。
最後の更新や検証、学習前後の評価、最終テスト、図の描画などにより、ノート全体は300秒を超える場合があります。
必ず8エポックまで比較する場合は、`MAX_TRAIN_SECONDS=None` にしてください。

```python
PATCH = 7
EPOCHS = 8
MAX_TRAIN_SECONDS = None  # 8エポックを完了する比較実験用
```

### 最終8エポック目で揃えた実測

2026年11月10日、作成者のPCで測定した各1シードの結果です。全て最終8エポック目の重みを使っています。

| PATCH | パッチ数 | テスト精度 | 学習ループ時間 | 学習バッチ数 |
|---|---:|---:|---|---:|
| 1 | 784 | 84.34% | 6時間31分55秒 | 1 |
| 2 | 196 | 85.87% | 33分12秒 | 16 |
| 4 | 49 | 85.51% | 5分31秒 | 32 |
| 7 | 16 | 86.26% | 1分58秒 | 32 |
| 14 | 4 | 87.18% | 43秒 | 32 |
| 28 | 1 | 86.13% | 36秒 | 32 |

時間は学習ループ中の検証を含み、最終テストや描画などは含みません。PATCH=1はチェックポイント保存も含み、
PATCH=2は最終重み取得のための再実行値です。実行環境やPC負荷によって時間は変わります。
バッチ数・更新回数・パラメータ数も異なるため、パッチサイズだけの厳密な比較ではありません。
この設定では14が最も高精度でしたが、一般に大きいパッチが優れるという結果ではありません。

### 評価モデルと保存ファイル

通常のテスト評価と図は、**検証損失が最小だったモデル `P`**を使います。
「8エポックまで学習した」ことと、「8エポック目の重みで評価した」ことは区別してください。
最終更新時点の重みは `P_last` に残すため、8エポック完了後に次の追加セルで確認できます。

```python
last_probs = predict_all(P_last, test_patches)
last_accuracy = np.mean(last_probs.argmax(-1) == raw_test_y)
print(f'Last update test accuracy: {last_accuracy:.2%}')
```

`tf02_results/patch_<PATCH>/` に、次のファイルを保存します。

- `tf02_fashion_model.npz`：検証損失最小のモデルと設定
- `tf02_fashion_model_last.npz`：最終更新時点のモデルと設定
- `tf02_metrics.json`：設定・学習履歴・実測時間・精度・混同行列
- `tf02_split_ids.npz`：学習・検証の画像ID

時間制限で途中終了した場合、lastは8エポック目とは限りません。保存設定の `completed_epochs` を確認してください。
学習済みモデル・ダウンロードデータ・過去の比較用JSONは、このリポジトリには含めていません。

## TF03 Fashion-MNISTをカタカナで答えるVision-Language Model

TF02のViTに、TF01の「1文字ずつ生成する」仕組みを足します。画像パッチの後ろに質問文「これは？」を置き、
続きの文字を生成して「スニーカー」「アンクルブーツ」のように答えます。

```text
画像パッチ → Linearで64次元へ ─┐
                                ├→ 1本の列 → 位置Embedding
質問と答えの文字 → Embedding表 ─┘
  → マスク付きSelf-Attention → SwiGLU MLP → 文字の位置から次の文字を予測
```

| TF02の小さなViT | TF03の小さなVLM |
|---|---|
| 入力は画像パッチ＋CLS | 入力は画像パッチ＋質問文＋答えの文字（CLSなし） |
| マスクなし | 画像は画像だけ、文字は画像全体と前の文字を参照 |
| CLSの出力から10クラスを選ぶ | 文字を1つずつ生成し、`<eos>` で止まる |
| 正解率＝クラスの一致 | 正解率＝生成した文字列の完全一致 |

初期設定は **PATCH=7、画像16トークン＋文字12位置の28トークン、語彙33、75,169パラメータ**です。
損失は答えの文字と `<eos>` だけで計算します。

### 実行する

1. 上表のTF03のColabボタンで開き、「ランタイム > すべてのセルを実行」を選びます。
2. データと分割はTF02と同じです（学習54,000枚・検証6,000枚・公式テスト10,000枚）。
3. 学習前後の生成結果、公式テストでの完全一致の正解率、混同行列（10語以外の出力を数える「その他」列つき）を確認します。
4. 1文字ずつの生成の確率と、答えの1文字目を出す位置から画像パッチへのAttentionを見ます。

図にカタカナを表示するため、日本語フォントがない環境（Colabなど）ではIPAexゴシックを実行時に取得します。
取得できない場合は、図のラベルだけ英語にして続行します。

### 実行結果の目安

手元のCPUで1回実行した結果です（SEED=0、8エポック、学習ループ約170秒）。環境と乱数で変わります。

| 項目 | 結果 |
|---|---:|
| 検証6,000枚の完全一致 | 86.33% |
| 公式テスト10,000枚の完全一致 | 86.38% |
| 10語以外の文字列を出力した数 | 0 / 10,000 |

間違いの中心はTF02と同じ、Tシャツ・シャツ・プルオーバー・コートの取り違えです。
画像の見分けという仕事は分類と同じなので、VLMにしても正解率は上がりません。

### 学習件数を変える

設定セルの `TRAIN_PER_CLASS`（1クラスあたりの学習枚数、最大5400）を変えると、学習データの件数と正解率の関係を調べられます。
検証画像は件数を変えても同じです。ノート末尾の任意セル（`RUN_SWEEP = True`）で、複数の件数をまとめて比較できます。

`tf03_results/patch_<PATCH>/` に、モデル（`tf03_vlm_model.npz`、`tf03_vlm_model_last.npz`）と `tf03_metrics.json` を保存します。

## 学習データの出典・ライセンス

- TF02・TF03は[Zalando Research / Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)を使用します。MIT License。
  実行時に公式データを取得し、公式READMEのMD5とIDX形式・件数を確認します。
- TF03は、日本語フォントがない環境でのみ [IPAexゴシック](https://moji.or.jp/ipafont/)（IPAフォントライセンス v1.0）を実行時に取得します。フォントはこのリポジトリには含めていません。

## ディレクトリ構成

```
vision/
├─ TF02_vision_transformer_fashion.ipynb      # 1ブロックViTで画像を分類
├─ TF03_vision_language_model_fashion.ipynb   # 1ブロックVLMでカタカナで答える
└─ README.md
```

追加のライブラリは不要です（`numpy`・`matplotlib` のみ。Colab には導入済み。ルートの `requirements.txt` と同じです）。
