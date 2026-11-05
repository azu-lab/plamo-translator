# 測定結果

README の推奨構成（vLLM + FP8、2B）を決めるときに取った測定です。手元の 1 台で取った値なので、目安として見てください。

## 測定環境

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop（VRAM 8 GB） |
| ドライバ | 566.24（Windows 側、CUDA 12.7 まで対応） |
| OS | Windows 11 + WSL2（Ubuntu 24.04）、CPU 24 スレッド |
| Transformers 側 | torch 2.6.0+cu126、transformers 5.22.1、bitsandbytes 0.50.2 |
| vLLM 側 | vLLM 0.32.0+cu129、torch 2.13.0+cu129 |
| 測定日 | 2026-11-05 |

モデルは次の 4 つの組み合わせを使いました。

- BF16 版: `pfnet/plamo-3-translate-2b`、`pfnet/plamo-3-translate-0.8b`
- FP8 版: `pfnet/plamo-3-translate-2b-FP8`、`pfnet/plamo-3-translate-0.8b-FP8`
- 4bit 版: BF16 版を `quantize.py`（bitsandbytes NF4、`lm_head` は BF16 のまま）で量子化したもの

## 1. 翻訳速度（64 段落）

Y. Takano, "Deadlock- and Starvation-free Formally Verified Client Library for Robots"（IEEE ICSRS 2023）の本文から 64 段落を取り出して測りました。

- 入力は合計約 3,100 トークン、出力は約 3,300 トークンです。
- 英語から日本語へ、常体（`form="plain"`）で、温度 0（貪欲法）で生成しました。
- 時間には、生成以外の処理（ウォームアップは除く）も含みます。

### 2B

| エンジン | 形式 | バッチ | 時間 | 生成速度 | VRAM（ピーク） |
|---|---|---|---|---|---|
| Transformers | 4bit | 1 | 154.6 秒 | 22 tok/s | 2.29 GiB |
| Transformers | 4bit | **8** | 40.1 秒 | 83 tok/s | 2.65 GiB |
| Transformers | 4bit | 16 | 44.4 秒 | 75 tok/s | 3.15 GiB |
| Transformers | 4bit | 32 | 66.6 秒 | 50 tok/s | 4.14 GiB |
| Transformers | BF16 | 1 | 143.1 秒 | 23 tok/s | 5.43 GiB |
| Transformers | BF16 | 8 | 40.7 秒 | 81 tok/s | 5.87 GiB |
| Transformers | BF16 | 16 | 46.3 秒 | 71 tok/s | 6.37 GiB |
| **vLLM** | **FP8** | 自動 | **5.6 秒** | **599 tok/s** | 確保量 0.75（約 6 GB） |
| vLLM | BF16 | — | 起動できず | — | — |

- vLLM の BF16 版は、重みだけで 5.64 GiB あり、KV キャッシュに回す分が足りずに起動しませんでした。メモリ使用率 0.85 でも 0.33 GiB しか残りませんでした。

### 0.8B

| エンジン | 形式 | バッチ | 時間 | 生成速度 | VRAM（ピーク） |
|---|---|---|---|---|---|
| Transformers | 4bit | 8 | 21.3 秒 | 155 tok/s | 1.16 GiB |
| Transformers | BF16 | 8 | 20.1 秒 | 166 tok/s | 1.93 GiB |
| **vLLM** | **FP8** | 自動 | **1.9 秒** | **1,771 tok/s** | 確保量 0.7 |

### 分かったこと

- **Transformers は、4bit でも BF16 でも速度がほぼ同じです。** 重みを 2.5 倍軽くしても速くならないので、遅さの原因は GPU の計算量ではありません。`generate` が 1 トークンごとに行う Python 側の処理が律速になっています。vLLM は CUDA Graph と、リクエストを自動でまとめる仕組み（continuous batching）でここを減らせるため、約 7〜11 倍速くなりました。
- **Transformers では、バッチは 8 が最も速く、16 以上では遅くなりました。** バッチの中で一番長い訳が終わるまで、ほかの文も待たされるためです。そのため、アプリの `--batch-size` の既定値は 8 にしています。
- **生成を止めるトークンは、vLLM の例と同じです。** `generate` は `eos_token_id=[1, 16]`（`<|plamo:bos|>`, `<|plamo:tag|>`）で止まり、モデル README の vLLM の例（`stop`）と一致していました。訳し終わったあとに余計な生成はしていません。

## 2. 論文 1 本を通しで翻訳した時間

アプリの PDF 翻訳タブで、HTML と Markdown を出力し、図を埋め込み、原文は併記しない設定で測りました。時間には PDF の解析と出力ファイルの作成も含みます。

| 構成 | ICSRS 2023（IEEE、6 ページ） | ECRTS 原稿（LIPIcs、図 14 枚） |
|---|---|---|
| 2B・Transformers・4bit（以前の既定） | 90 秒 | 134 秒 |
| **2B・vLLM・FP8（推奨）** | **15 秒** | **21 秒** |
| 0.8B・vLLM・FP8 | 9 秒 | 12 秒 |
| 0.8B・Transformers・4bit | 60 秒 | — |

## 3. メモリ

| 構成 | GPU（nvidia-smi） | RAM |
|---|---|---|
| 2B・vLLM・FP8（`GPU_UTIL=0.7`） | 5.9 GB | vLLM 3.6 GB + アプリ 0.4 GB |
| 0.8B・vLLM・FP8（`GPU_UTIL=0.7`） | 6.0 GB | — |
| 0.8B・vLLM・FP8（`GPU_UTIL=0.5`） | 4.5 GB（KV キャッシュ 375,005 トークン分） | — |
| 2B・Transformers・4bit | 2.5〜2.6 GB | 約 1.8 GB |
| 2B・Transformers・BF16 | 5.9 GB | 約 1.7 GB |

- **vLLM は起動時に「GPU メモリ全体 × `GPU_UTIL`」を先に確保します。** そのため、0.8B でも 2B と同じくらい GPU メモリを使っています。0.8B なら `GPU_UTIL=0.5` に下げても足ります（重みは 1.4 GiB）。
- **2B・FP8・`GPU_UTIL=0.7` の場合、KV キャッシュは 35,184 トークン分でした。** 論文の段落を数十件並列に処理するには十分です。
- **Transformers + bitsandbytes では、何も対策しないと RAM を約 8 GB 余分に使いました。** bitsandbytes を import した時点で `torch.compile` 用のワーカーが 24 個起動していたためです。今は `TORCHINDUCTOR_COMPILE_THREADS=1` を設定して止めており、上の表の値はその状態のものです。

## 訳の質

上の 64 段落について、**2B・BF16（Transformers）の出力を基準**にして、各構成の出力と比べました。

- 完全一致: 訳文が基準と 1 文字も違わない段落の数
- 類似度: `difflib.SequenceMatcher` の比率の平均（1.0 で完全一致）

| 構成 | 2B BF16 との完全一致 | 2B BF16 との類似度 | 0.8B BF16 との類似度 |
|---|---|---|---|
| 2B・FP8（vLLM） | **50/64** | **0.992** | 0.892 |
| 2B・4bit（Transformers） | 34/64 | 0.968 | 0.896 |
| 0.8B・BF16（Transformers） | 13/64 | 0.892 | 1.000 |
| 0.8B・FP8（vLLM） | 13/64 | 0.886 | 0.979 |
| 0.8B・4bit（Transformers） | 13/64 | 0.890 | 0.962 |

- **FP8 版（PFN の公式配布）は、bitsandbytes の 4bit よりも元のモデルに近い訳になりました。** 違いが出た段落も、語順が入れ替わる程度でした。

  ```
  BF16: …通常は、レスポンスを更新して応答を送信するが、クラッシュをエミュレートするためにexitを呼び出している（6行目）。
  FP8 : …通常は、レスポンスを更新して応答を送信するが、ここではexitを呼び出してクラッシュをエミュレートしている（6行目）。
  ```

- **0.8B と 2B の違いは、量子化による違いよりずっと大きいです。** 0.8B は固有名詞や専門用語を普通の言葉として訳してしまうことがあります。

  ```
  原文: In this paper, safe drive and async lib are proposed as new libraries. …
  2B  : 本論文では、新たなライブラリとしてsafe driveおよびasync libを提案する。…
  0.8B: 本論文では、安全なドライブおよび非同期ライブラリを新たなライブラリとして提案する。…

  原文: RELATED WORK
  2B  : 関連研究
  0.8B: 関連作品
  ```

類似度は、2B・BF16 にどれだけ近いかを測った指標で、訳の正しさそのものではありません。また、評価に使ったのは 1 本の論文だけです。

## vLLM を動かすまでにつまずいたこと

`requirements-vllm.txt` と `serve.sh` の設定は、次の問題を避けるためのものです。

1. **最新の vLLM（PyPI の 0.33.0）は CUDA 13 版の torch 2.14 を要求し、ドライバ 566.24（CUDA 12.7）では `The NVIDIA driver on your system is too old` で動きませんでした。**
   - 0.33.0 には CUDA 12 系の wheel がありません。
   - 1 つ前の 0.32.0 には、GitHub のリリースに `+cu129` 版があり、PLaMo 3（`vllm/model_executor/models/plamo3.py`）にも対応しています。
   - CUDA 12.x の互換の仕組みにより、CUDA 12.9 のランタイムが 12.7 のドライバで動きました。
   - ドライバを CUDA 13 対応の版に更新すれば、0.33.0 も使えるはずです（確認していません）。
2. **PyPI の `torchcodec`（CUDA 13 版）が、vLLM の import 時に `libnvrtc.so.13` が無いと言って失敗しました。**
   - 翻訳には使わない、音声・動画用のライブラリです。
   - PyTorch の配布元にある `torchcodec==0.16.0+cu129` に固定しています。
3. **FlashInfer のサンプラーが、初回にカーネルを JIT ビルドしようとして `Could not find nvcc` で止まりました。**
   - 温度 0（貪欲法）ではこのサンプラーは不要です。
   - `serve.sh` で `VLLM_USE_FLASHINFER_SAMPLER=0` を設定し、CUDA Toolkit（nvcc）なしで起動できるようにしています。
4. **vLLM は、アプリ側と版がぶつかります（transformers 5.20 未満、torch 2.13 が必要）。** そのため、仮想環境を分け（`.venv-vllm`）、アプリとは HTTP でやり取りする構成にしています。
5. **vLLM の起動時に、テンプレートのエラーが 1 行出ます（`jinja2.exceptions.TemplateError`）。**
   - PLaMo のチャットテンプレートは `target_lang` が必須なので、vLLM が起動時に翻訳オプションなしで試しに読み込むと失敗するようです。
   - 警告だけで、翻訳には影響しませんでした。

## 再現のしかた

測定に使ったスクリプトはリポジトリには入れていません。おおまかな手順は次のとおりです。

- **Transformers**: `backends.TransformersBackend` と同じ方法で生成しました。`apply_chat_template` を `padding=True` で呼び、`generate(do_sample=False)` で生成しています。段落は入力の長さ順に並べ、バッチ単位で処理しました。
- **vLLM**: `vllm.LLM(max_model_len=8192, max_num_seqs=64)` に 64 段落を一度に渡し、`SamplingParams(temperature=0, stop_token_ids=[1, 16])` で生成しました。
- **時間**: ウォームアップ（4 段落）のあとに測りました。生成トークン数は、終端トークンとパディングを除いて数えています。
