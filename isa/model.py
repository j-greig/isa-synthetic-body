"""Thin wrapper around a Hugging Face causal LM with activation capture and steering.

Uses plain PyTorch forward hooks, so it works on Mac (MPS), CUDA and CPU without
extra interpretability libraries.
"""
from __future__ import annotations

import contextlib
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(dtype: str, device: str):
    if dtype == "auto":
        return torch.float32 if device == "cpu" else torch.bfloat16
    return getattr(torch, dtype)


class LM:
    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto"):
        self.device = pick_device(device)
        self.dtype = pick_dtype(dtype, self.device)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=self.dtype, attn_implementation="eager"
        ).to(self.device).eval()
        # Avoid the per-call "max_new_tokens and max_length both set" warning (we always pass max_new_tokens).
        self.model.generation_config.max_length = None
        self.layers = self._find_layers()
        self._steer = {}   # layer -> callable(hidden) -> delta

    def _find_layers(self):
        m = self.model
        for path in ("model.layers", "model.language_model.layers", "transformer.h", "gpt_neox.layers"):
            obj = m
            try:
                for p in path.split("."):
                    obj = getattr(obj, p)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("Could not find decoder layers on this model")

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    # ---- prompts --------------------------------------------------------
    def chat_ids(self, messages, add_generation_prompt=True) -> torch.Tensor:
        ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, return_tensors="pt",
            return_dict=True,
        )["input_ids"]
        return ids.to(self.device)

    # ---- hooks ----------------------------------------------------------
    @staticmethod
    def _hidden(out):
        return out[0] if isinstance(out, tuple) else out

    @staticmethod
    def _replace(out, h):
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    @contextlib.contextmanager
    def capture(self, layers):
        """Record each layer's output (residual stream after the layer)."""
        store, handles = {}, []
        for L in layers:
            def hook(mod, inp, out, L=L):
                store[L] = self._hidden(out).detach()
            handles.append(self.layers[L].register_forward_hook(hook))
        try:
            yield store
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def steer(self, layer: int, vector: torch.Tensor | None, scale: float):
        """Add scale*vector to the residual stream at `layer` on every position."""
        if vector is None or scale == 0:
            yield
            return
        v = vector.to(self.device, self.dtype)

        def hook(mod, inp, out):
            h = self._hidden(out)
            return self._replace(out, h + scale * v)

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    # ---- generation & measurement --------------------------------------
    @torch.no_grad()
    def generate(self, messages, max_new_tokens=60, temperature=0.8, seed=None):
        """Return (text, mean entropy in nats over generated tokens)."""
        if seed is not None:
            torch.manual_seed(seed)
        ids = self.chat_ids(messages)
        out = self.model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None, top_p=0.95,
            output_logits=True, return_dict_in_generate=True,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        gen = out.sequences[0, ids.shape[1]:]
        ents = []
        for lg in out.logits:
            p = torch.softmax(lg[0].float(), -1)
            ents.append(float(-(p * torch.log(p + 1e-12)).sum()))
        text = self.tok.decode(gen, skip_special_tokens=True).strip()
        return text, (sum(ents) / len(ents) if ents else float("nan"))

    @torch.no_grad()
    def reply_hidden(self, messages, reply: str, layers):
        """Hidden states over the reply tokens only. Returns {layer: [n_tokens, d]} (float32, cpu)."""
        prompt = self.chat_ids(messages)
        r = self.tok(reply, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        if r.shape[1] == 0:
            r = self.tok(".", add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        full = torch.cat([prompt, r], 1)
        with self.capture(layers) as st:
            self.model(full)
        n = prompt.shape[1]
        return {L: st[L][0, n:].float().cpu() for L in layers}

    @torch.no_grad()
    def pre_state(self, messages, layers):
        """One forward pass over the prompt (with the generation prompt). Returns
        (hidden at the last prompt position {layer: [d]}, surprisal of the last user message).

        Surprisal = mean negative log-likelihood (nats/token) of the last message's tokens:
        the model's own prediction error for its input, a signal it never sees as text."""
        ids = self.chat_ids(messages)
        with self.capture(layers) as st:
            logits = self.model(ids).logits[0].float()
        pre = {L: st[L][0, -1].float().cpu() for L in layers}
        content = self.tok(messages[-1]["content"], add_special_tokens=False)["input_ids"]
        seq, n = ids[0].tolist(), len(content)
        start = None
        for i in range(len(seq) - n, max(-1, len(seq) - n - 40), -1):
            if seq[i:i + n] == content:
                start = i
                break
        if start is None or start == 0 or n == 0:
            return pre, float("nan")
        lp = torch.log_softmax(logits[start - 1:start + n - 1], -1)
        tgt = torch.tensor(content, device=lp.device)
        nll = -lp.gather(1, tgt[:, None]).squeeze(1)
        return pre, float(nll.mean())

    @torch.no_grad()
    def next_token_logprobs(self, messages, words):
        """Log-probability that the model's answer starts with each word."""
        ids = self.chat_ids(messages)
        logits = self.model(ids).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        res = {}
        for w in words:
            cands = set()
            for form in (w, w.capitalize(), " " + w):
                t = self.tok(form, add_special_tokens=False)["input_ids"]
                if t:
                    cands.add(t[0])
            res[w] = float(torch.logsumexp(lp[list(cands)], 0))
        return res

    def expected_digit(self, messages, lo=0, hi=9):
        """Expected value of a single-digit answer (lo..hi), from next-token probabilities."""
        lps = self.next_token_logprobs(messages, [str(i) for i in range(lo, hi + 1)])
        ps = {int(k): math.exp(v) for k, v in lps.items()}
        z = sum(ps.values())
        return sum(k * p for k, p in ps.items()) / z if z > 0 else float("nan")
