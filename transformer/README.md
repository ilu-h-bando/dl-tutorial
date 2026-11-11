# Transformer ハンズオン教材

Deep Learning 講義の **「Transformer」パート（P21〜39）** の部品を、**NumPyで1つずつ組み立てて動かす**チュートリアルです。
TF01は1ブロックのGPTで和歌の続きを生成し、TF02はその部品を使った小さなVision Transformerで服・靴・バッグなどを分類します。
PyTorchやTensorFlow、学習済みモデルは使いません。Google Colabのボタンからブラウザで実行できます。

## ノートブック

| ノートブック | 内容 | データ | Colab |
|---|---|---|---|
| **[TF01_tiny_gpt_waka.ipynb](TF01_tiny_gpt_waka.ipynb)** | NumPy で 1 段の GPT（約 7 万パラメータ）を実装。八代集の和歌を学習し、最初の 5 音（または 5＋7 音）から「五・七・五・七・七」の歌を生成する | 実行時に Wikisource から自動取得 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilu-h-bando/dl-tutorial/blob/main/transformer/TF01_tiny_gpt_waka.ipynb) |
| **[TF02_vision_transformer_fashion.ipynb](TF02_vision_transformer_fashion.ipynb)** | NumPyで1ブロックのViTを実装。パッチ化・CLS・位置情報・Attentionを観察し、Fashion-MNISTを10分類する | 実行時にFashion-MNIST公式GitHubから自動取得 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilu-h-bando/dl-tutorial/blob/main/transformer/TF02_vision_transformer_fashion.ipynb) |

## TF01 和歌GPTで体験すること

- 学習前は文字がでたらめに並ぶ → 学習すると、五七五七七の形をした和歌のような並びが出てくる
- 区切り記号「／」もふつうの 1 文字として学習させるだけで、**音数のリズムまで学べる**ことを、音数の自動判定で確かめる
- 歌の**最初の部分を入力すると、その続きを詠む**（入力文を和歌に*変換*するのではありません）
- 逆伝播は数値微分と照らして、正しいことを確認する（勾配チェック）
- Attention の図で、モデルが区切り記号などに注目している様子を見る

> 1 段・約 7 万パラメータ・約 20 万文字の学習なので、出てくるのは「和歌らしい形と言葉の並び」です。意味の通る名歌にはなりません。

## TF01と講義との対応

| 講義のページ | ノートの節 |
|---|---|
| P24〜25 入力エンコーディング・Dense Vector | 2. トークン化と Embedding |
| P26 ポジション・エンコーディング | 3. 位置情報を足す |
| P27〜28 Self-Attention（Q・K・V） | 4. Self-Attention |
| P30 Masked Attention | 5. 未来を見せない（マスク） |
| P29 Multi-Head Attention | 6. 複数のヘッド |
| P31 Feed Forward（Up・Gate・Down） | 7. MLP（SwiGLU） |
| P34〜35 Residual・LayerNorm | 8. 残差接続と LayerNorm |
| P36〜37 Linear・Softmax | 9. 出力層（Linear → Softmax） |
| P32〜33 Transformer の直列化 | 10. 部品を組み立てる（1 段だけ） |

## TF01の使い方

### 1. Colab で開く
上表の「Open in Colab」バッジをクリックします（初回はブラウザで Google にログイン）。
このリポジトリは公開なので、GitHub の認証やトークンは不要です。

### 2. 実行する
メニューの「ランタイム > すべてのセルを実行」を押します。
実行中にインターネットから和歌のデータを取得するので、ネットワークへの接続が必要です。
学習は CPU で動かせます（手元の PC では約 2 分でした）。

### 3. 自分で入力して試す
ノートの途中に**入力欄**のセルがあります。実行すると入力欄が出るので、歌の最初の部分を**ひらがな**で入力して Enter を押すと、続きを詠みます。
何も入力せずに Enter を押すと終了します。

- 「はるかすみ」（5 音）、「はるかすみたてるやいつこ」（5＋7 音）のように入力します（5・7 音ごとの区切りは自動で入ります）
- 学習データの和歌は**昔の表記**です。**濁点・半濁点は書かれていません**（「かぜ」→「かせ」）。入力した濁点は自動で外し、カタカナはひらがなに直します
- 仮名遣いも昔のものです。現代の発音どおりに書くと、学習データにない並びになることがあります（例：「きょう」→「けふ」）
- 漢字は入力できません

### 4. いじってみる
ノート末尾の「やってみよう」で、設定を変えて挙動の違いを体感できます。たとえば次のとおりです。

| 設定 | 試すこと |
|---|---|
| `ALIGN_TO_POEM = False` | 学習データの切り出しを「歌の頭から」にそろえない。音数が合う歌の割合が下がる（位置エンコーディングが効いていることの確認） |
| `FORCE_RHYTHM = True`（入力欄） | 音数を必ず 5・7・5・7・7 にそろえる |
| `DROPOUT = 0.0` と `STEPS = 6000` | ドロップアウトなしで長く学習する。過学習・暗記の様子を見る |
| `USE_MASK = False` | 未来を見せないマスクを外す。何が起きるか見る |
| `D = 128`, `D_FF = 512` | モデルを大きくする |

## TF01の実行結果の目安

学習に使っていない歌の最初の 5 音から 565 首を詠ませて、音数が 5・7・5・7・7 に合った割合を数えました（手元の PC での 1 回の実行結果です）。

| 条件 | 音数が合った割合 |
|---|---|
| 学習データを歌の頭から切り出す（`ALIGN_TO_POEM = True`、既定） | 100% |
| 文字列のどこからでも切り出す（`ALIGN_TO_POEM = False`） | 約 70% |

どちらも、学習データの歌と完全に同じ歌が出た割合は 0% でした。

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

## 学習データの出典・ライセンス

- [Wikisource](https://ja.wikisource.org/) の『古今和歌集』『後撰和歌集』『拾遺和歌集』『千載和歌集』『新古今和歌集』（各歌に付いている、五・七・五・七・七で区切ったひらがなの読みを使用）
- 作品は著作権が切れています。ページの文章のライセンスは [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ja)（Wikisource の寄稿者による）です
- TF01は**実行時に Wikisource の API から取得するだけ**で、データをこのリポジトリには含めていません

- TF02は[Zalando Research / Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)を使用します。MIT License。
  実行時に公式データを取得し、公式READMEのMD5とIDX形式・件数を確認します。

## ディレクトリ構成

```
transformer/
├─ TF01_tiny_gpt_waka.ipynb              # 1段GPTで和歌を詠む
├─ TF02_vision_transformer_fashion.ipynb # 1ブロックViTで画像を分類
└─ README.md
```

追加のライブラリは不要です（`numpy`・`matplotlib` のみ。Colab には導入済み。ルートの `requirements.txt` と同じです）。
