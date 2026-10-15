"""Optional Gemma Scope 2 SAE features via sae_lens."""
from __future__ import annotations

import torch


class SAEReader:
    def __init__(self, release: str, sae_id: str, device: str = "cpu"):
        from sae_lens import SAE
        sae = SAE.from_pretrained(release, sae_id, device=device)
        self.sae = sae[0] if isinstance(sae, tuple) else sae
        self.sae.eval()
        self.device = device

    @torch.no_grad()
    def encode_mean(self, h: torch.Tensor) -> torch.Tensor:
        """h: [n_tokens, d] -> mean feature activations [n_features]."""
        f = self.sae.encode(h.to(self.device, torch.float32))
        return f.mean(0).cpu()

    def top(self, h: torch.Tensor, k: int = 10):
        m = self.encode_mean(h)
        v, i = m.topk(k)
        return [(int(a), round(float(b), 4)) for a, b in zip(i, v)]
