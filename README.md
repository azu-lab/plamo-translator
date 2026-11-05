# PLaMo Translator

[PLaMo 3 Translate](https://huggingface.co/collections/pfnet/plamo-3-translate)（Preferred Networks）を手元の GPU で動かす、ローカル翻訳 GUI です。

- **テキスト翻訳**: 文章を貼り付けて翻訳します。翻訳元は自動判定にでき、翻訳先は16言語、文体は常体と敬体から選べます。
- **PDF 論文翻訳**: 論文の章立て（`1` / `2.1` / `III.` / `A.` など）を保ったまま全文を訳し、**HTML（目次付き）と Markdown** で出力します。
  - 著者・所属と参考文献は原文のまま残します。
  - 図と表はそのまま埋め込みます。
  - 段落ごとに原文を併記することもできます（任意）。
- **4bit 量子化**: モデルを 5.4 GB から 2.2 GB に縮め、VRAM 約 2.5 GB で動かします。

翻訳はすべてローカルで行います。PDF や文章を外部に送ることはありません。

## 動作確認した環境

| 項目 | 内容 |
|---|---|
| OS | Windows 11 + WSL2（Ubuntu 24.04） |
| GPU | NVIDIA GeForce RTX 4060 Laptop（VRAM 8 GB）、Windows 側ドライバ 566.24 |
| Python | 3.12 |
| モデル | `pfnet/plamo-3-translate-2b`（revision `355637cc`） |

上記の環境で測った値は次のとおりです。

| モデル | ファイルサイズ | 翻訳中の VRAM | 6ページの論文の翻訳時間 |
|---|---|---|---|
| 2B BF16（`./model`） | 5.4 GB | 約 5.9 GB | — |
| 2B 4bit（`./model-4bit`、既定） | 2.2 GB | 約 2.6 GB | 約 90 秒 |

Linux（ネイティブ）でも同じ手順で動くはずですが、確認していません。

## リポジトリに含まれないもの

次のものは `.gitignore` でリポジトリから外しています。各自が下の手順で作成してください。

| パス | 中身 | 作り方 |
|---|---|---|
| `.venv/` | Python 仮想環境 | [手順 2](#2-python-環境を作る) |
| `model/` | PLaMo 3 Translate 2B の重み（BF16、5.4 GB） | [手順 3](#3-モデルをダウンロードする) |
| `model-4bit/` | 上の重みを 4bit 量子化したもの（2.2 GB） | [手順 4](#4-モデルを-4bit-量子化する推奨) |

モデルの重みは [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-ja) で公開されています。そのため、このリポジトリでは再配布せず、各自が Hugging Face から取得する形にしています。

## セットアップ

以下のコマンドは、すべて **WSL2 の Ubuntu のターミナル**で実行します。

### 1. WSL2 から GPU が見えるか確認する

```bash
nvidia-smi
```

GPU 名（例: `NVIDIA GeForce RTX 4060 ...`）が表示されれば問題ありません。

- WSL2 では **Windows 側の NVIDIA ドライバ**を使います。Ubuntu 側に Linux 用のドライバを入れる必要はありません（入れないでください）。
- `nvidia-smi: command not found` になる場合は、Windows 側の NVIDIA ドライバを更新し、PowerShell で `wsl --update` を実行してください。
- 詳しくは [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/) を参照してください。

### 2. Python 環境を作る

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-dev git build-essential

git clone <このリポジトリの URL> plamo-translator
cd plamo-translator

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` には **CUDA 12.6 版の PyTorch** を取得する設定（`--extra-index-url`）が入っています。そのため、PyTorch を別途インストールする必要はありません。ダウンロード量は 3 GB 程度です。

インストールが終わったら、GPU を使えるか確認します。

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

`CUDA: True` と表示されれば問題ありません。`False` になる場合は、手順 1 に戻ってください。

> **ファイルの置き場所**: リポジトリは `/mnt/c/...`（Windows のドライブ）ではなく、`~/` の下など WSL2 側のファイルシステムに置いてください。`/mnt/c` 経由だとモデルの読み込みが大幅に遅くなります。

### 3. モデルをダウンロードする

先に [ライセンス](https://plamo.preferredai.jp/info/plamo-community-license-ja) を確認してください。

```bash
hf download pfnet/plamo-3-translate-2b --local-dir ./model
```

約 5.4 GB あります。動作を確認した版と同じものを使いたい場合は、`--revision 355637cc04db67195ba87a97da85eaf1d52a22e8` を付けてください。

### 4. モデルを 4bit 量子化する（推奨）

```bash
python quantize.py
```

`./model` を読み込み、4bit（NF4）に量子化したモデルを `./model-4bit` に保存します。GPU で 1 分ほどかかり、最初の 1 回だけ実行すれば済みます。

- 8bit にしたい場合は `python quantize.py --bits 8` を実行してください（保存先は `./model-8bit`）。
- 量子化したあとも `./model` は残しておくと、BF16 版と訳の質を比べられます。不要なら削除してかまいません。

### 5. 起動する

```bash
source .venv/bin/activate   # 新しいターミナルを開いたとき
python app.py
```

起動後、Windows のブラウザで <http://localhost:7860> を開きます（WSL2 から Windows への localhost 転送は自動で行われます）。

## 使い方

### テキスト翻訳タブ

1. 翻訳元・翻訳先・文体を選びます。
2. 原文を入力し、「翻訳」を押します。

「⇄ 言語を入れ替え」を押すと、翻訳の向きと原文・訳文が入れ替わります。

### PDF 翻訳タブ

1. PDF を選び（複数可）、出力形式（HTML / Markdown）を選んで「翻訳開始」を押します。
2. 結果のファイルをダウンロードします。複数の PDF を入れた場合は ZIP にもまとめます。

PDF 翻訳タブのオプションは次のとおりです。

| オプション | 内容 |
|---|---|
| 段落ごとに原文を併記 | 各段落の下に、折りたたんだ原文を付けます |
| 図を埋め込む | 図を画像としてファイルに埋め込みます（ファイルが数百 KB〜数 MB 大きくなります） |
| 参考文献も翻訳する | 既定では参考文献は原文のまま残します |

### 起動オプション

| オプション | 既定値 | 内容 |
|---|---|---|
| `--model PATH` | `./model-4bit`（なければ `./model`） | 使うモデル。環境変数 `PLAMO_MODEL` でも指定できます |
| `--port N` | `7860` | 待ち受けるポート |
| `--batch-size N` | `8` | PDF 翻訳で一度に訳す段落数。VRAM が足りなければ下げてください |

例:

```bash
python app.py --model ./model            # 量子化しない BF16 版で起動する
python app.py --batch-size 4 --port 7861
```

## ほかのサイズのモデルを使う

PLaMo 3 Translate には 0.8B / 2B / 8B / 31B の 4 サイズがあります。Transformers で読み込める BF16 版であれば、`--model` で切り替えられます。

```bash
hf download pfnet/plamo-3-translate-0.8b --local-dir ./model-0.8b
python quantize.py --src ./model-0.8b --dst ./model-0.8b-4bit   # 任意
python app.py --model ./model-0.8b-4bit
```

- 2B 以外のサイズでは動作を確認していません。
- 8B を 8 GB の GPU で動かすには、4bit 量子化が必要になる見込みです。
- Hugging Face にある `-FP8` / `-W4A16` / `-GGUF` などの配布版は、vLLM や llama.cpp 向けの形式です。このアプリ（Transformers + bitsandbytes）では読み込めません。

## ファイル構成

```
plamo-translator/
├── app.py            # GUI（Gradio）とモデルの読み込み・翻訳
├── paper.py          # PDF → 章立ての抽出 → Markdown / HTML 出力
├── quantize.py       # モデルの 4bit / 8bit 量子化
├── requirements.txt
├── model/            # (git 管理外) ダウンロードしたモデル
└── model-4bit/       # (git 管理外) 量子化したモデル
```

## 制限事項

- **スキャンした PDF（画像だけの PDF）は翻訳できません。** OCR は使っていません。
- **見出しが抜けることがあります。** 章立ては PDF の文字の大きさから判定しています（[pymupdf4llm](https://github.com/pymupdf/pymupdf4llm)）。見出しを取り出せなかった章は、前の章の中に入ります。
- **数式は崩れることがあります。** PDF から取り出した文字をそのまま翻訳しているためです。
- **訳語が章をまたいで揺れることがあります。** 速度を優先して段落を数個ずつまとめて翻訳しており、前の段落の訳は参照していません。
- **1 語だけの見出し（`Rust`、`GEDF` など）は原文のまま出します。** 文脈なしで訳すと誤訳しやすいためです。
- **出力は機械翻訳です。** 内容は必ず原文で確認してください。

## トラブルシューティング

### アプリを終了しても、Windows のタスクマネージャーでメモリが減らない

WSL2 は、Linux 側で解放したメモリをすぐには Windows に返しません（タスクマネージャーでは `vmmemWSL` / `Vmmem` として表示されます）。

- すぐに解放したい場合は、PowerShell で `wsl --shutdown` を実行します。
- 恒久的に対策したい場合は、`C:\Users\<ユーザー名>\.wslconfig` に以下を書き、`wsl --shutdown` で WSL を再起動します。

  ```ini
  [wsl2]
  memory=16GB

  [experimental]
  autoMemoryReclaim=gradual
  ```

### 起動しただけで RAM を 8 GB 以上使う

bitsandbytes を import すると、PyTorch が CPU のコア数ぶん `torch.compile` 用のワーカーを起動します。24 コアの環境では、これで約 8 GB を使いました。

`app.py` と `quantize.py` では、先頭で `TORCHINDUCTOR_COMPILE_THREADS=1` を設定してこれを止めています。自分のスクリプトから bitsandbytes を使うときも、`import torch` の前に同じ設定をしてください。

### `CUDA out of memory`

- `--batch-size` を下げてください（例: `--batch-size 2`）。
- BF16 版（`./model`）を使っている場合は、4bit 版に切り替えてください。

## ライセンス

- このリポジトリのコード: （ライセンスを決めて記載してください）
- PLaMo 3 Translate のモデル: [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-ja)
  - 個人利用と非商用利用（学術利用を含む）は、登録なしで無償で利用できます。
  - 商用利用には条件があります。詳しくはライセンス本文と [公開時の告知](https://www.preferred.jp/ja/blog/tech/plamo-3-translate-weight-release) を確認してください。
