"""翻訳エンジン

- VllmBackend:         別プロセスで起動した vLLM サーバー（OpenAI 互換 API）に投げる。推奨・高速
- TransformersBackend: このプロセスにモデルを読み込んで Transformers で生成する。vLLM が使えない環境向け

どちらも translate_many(texts, source_lang, target_lang, form, progress) で
訳文のリストを（入力と同じ順番で）返す。
"""

import gc
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


def template_kwargs(source_lang, target_lang, form):
    """チャットテンプレートに渡す翻訳オプション（model/README_ja.md 参照）"""
    kwargs = dict(target_lang=target_lang)
    if source_lang and source_lang != "auto":
        kwargs["source_lang"] = source_lang
    # form は target_lang="ja" のときだけ指定できる
    if target_lang == "ja" and form:
        kwargs["form"] = form
    return kwargs


# ------------------------------------------------------------
# vLLM
# ------------------------------------------------------------
class VllmBackend:
    # PLaMo の PFML3 形式で、訳文の終わりを表すトークン
    STOP = ["<|plamo:tag|>", "<|plamo:bos|>"]

    def __init__(self, base_url, max_workers=32, timeout=600):
        self.base_url = base_url.rstrip("/")
        self.max_workers = max_workers
        self.client = httpx.Client(timeout=timeout)
        models = self.client.get(f"{self.base_url}/models").json()["data"]
        self.model = models[0]["id"]
        self.name = f"vLLM ({self.model} @ {self.base_url})"

    @staticmethod
    def is_available(base_url):
        try:
            return httpx.get(f"{base_url.rstrip('/')}/models", timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    def _translate_one(self, text, kwargs):
        # 訳文の長さは原文と同程度。文字数はトークン数以上になるので上限の目安に使う
        max_tokens = min(2048, max(256, len(text)) + 128)
        r = self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": text}],
                "chat_template_kwargs": kwargs,
                "temperature": 0,
                "max_tokens": max_tokens,
                "stop": self.STOP,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def translate_many(self, texts, source_lang, target_lang, form, progress=None):
        kwargs = template_kwargs(source_lang, target_lang, form)
        results = [None] * len(texts)
        # vLLM はサーバー側で受け取ったリクエストをまとめて処理するので、並列に投げるだけでよい
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._translate_one, t, kwargs): i for i, t in enumerate(texts)}
            for done, future in enumerate(as_completed(futures), 1):
                results[futures[future]] = future.result()
                if progress:
                    progress(done, len(texts))
        return results

    def free_memory(self):
        pass


# ------------------------------------------------------------
# Transformers
# ------------------------------------------------------------
class TransformersBackend:
    def __init__(self, model_path, batch_size=8, max_batch_tokens=4096):
        # bitsandbytes は import 時に torch.compile を呼ぶため、PyTorch が
        # CPUコア数ぶん（24個・計8GB程度）のコンパイル用ワーカーを先に起動してしまう。
        # GPU推論では使わないので、torch を import する前に無効化しておく。
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # バッチ翻訳では左側をパディングする（モデルの README に従う）
        self.tokenizer.padding_side = "left"

        # 元の重みが bf16 なので bf16 のまま読み込む
        # （fp16/fp32 への変換は変換用の一時メモリが必要になり、fp32 だとサイズも2倍になる）
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
            device_map=device,
        ).eval()

        self.name = f"Transformers ({model_path} on {device})"
        if torch.cuda.is_available():
            print(f"VRAM: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    def translate_batch(self, texts, source_lang, target_lang, form):
        """複数の文章をまとめて1回で生成する"""
        torch = self.torch
        inputs = self.tokenizer.apply_chat_template(
            [[{"role": "user", "content": t}] for t in texts],
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs(source_lang, target_lang, form),
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]
        # 出力長を入力に合わせて抑え、KVキャッシュの確保量を減らす
        max_new_tokens = min(2048, input_len * 2 + 128)

        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

        results = self.tokenizer.batch_decode(output[:, input_len:], skip_special_tokens=True)
        del inputs, output
        return [r.strip() for r in results]

    def translate_many(self, texts, source_lang, target_lang, form, progress=None):
        """長さの近い段落どうしでバッチにして翻訳する（順番は元のまま返す）"""
        lengths = [len(self.tokenizer(t)["input_ids"]) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: lengths[i])

        batches, current = [], []
        for i in order:
            # 長さ順なので、最後に足す要素がバッチ内で一番長い
            if current and (len(current) >= self.batch_size
                            or lengths[i] * (len(current) + 1) > self.max_batch_tokens):
                batches.append(current)
                current = []
            current.append(i)
        if current:
            batches.append(current)

        results = [None] * len(texts)
        done = 0
        try:
            for batch in batches:
                translated = self.translate_batch(
                    [texts[i] for i in batch], source_lang, target_lang, form
                )
                for i, r in zip(batch, translated):
                    results[i] = r
                done += len(batch)
                if progress:
                    progress(done, len(texts))
        finally:
            self.free_memory()
        return results

    def free_memory(self):
        """推論で使った一時バッファを解放する"""
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
