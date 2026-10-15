"""Phase 1b: label SAE features locally (no web), by the tokens that activate them most.

Runs a small corpus (probe passages, held-out passages, scenario lines) through the model,
encodes each token with the SAE, and prints the top-activating tokens with context for
each feature ID. Writes docs/features_L<layer>.md.

  python scripts/01b_label_features.py --layer 13 --features 2404,278,743,896
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from isa.cli import parser, config
from isa.model import LM
from isa.sae import SAEReader
from isa import passages
from isa.scenarios import SCENARIOS, AGENT_SCENARIOS

ap = parser(__doc__)
ap.add_argument("--layer", type=int, default=13)
ap.add_argument("--features", required=True, help="comma-separated feature IDs")
ap.add_argument("--top", type=int, default=8)
args = ap.parse_args()
cfg = config(args)
feats = [int(x) for x in args.features.split(",")]

corpus = [it["text"] for it in passages.build(per_ctx=2)]
try:
    from isa import heldout
    corpus += [it["text"] for it in heldout.build()]
except Exception:
    pass
for sc in (SCENARIOS, AGENT_SCENARIOS):
    for s in sc.values():
        for fr in s.values():
            corpus.append(fr["setup"])
            for ph in ("rest", "rise", "peak", "recovery"):
                corpus += fr[ph]

lm = LM(cfg.model_id, cfg.device, cfg.dtype)
sae = SAEReader(cfg.sae_release, cfg.sae_id_template.format(layer=args.layer))
prompt = [{"role": "user", "content": passages.PROMPT}]

hits = defaultdict(list)  # feature -> [(act, context)]
for text in corpus:
    h = lm.reply_hidden(prompt, text, [args.layer])[args.layer]
    with torch.no_grad():
        f = sae.sae.encode(h.to(sae.device, torch.float32)).cpu()
    toks = lm.tok(text, add_special_tokens=False)["input_ids"]
    for fid in feats:
        col = f[:, fid]
        i = int(col.argmax())
        if col[i] > 0 and i < len(toks):
            left = lm.tok.decode(toks[max(0, i - 6):i])
            tok = lm.tok.decode(toks[i:i + 1])
            right = lm.tok.decode(toks[i + 1:i + 4])
            hits[fid].append((float(col[i]), f"{left}[[{tok}]]{right}"))

lines = [f"# SAE features, layer {args.layer}\n",
         f"SAE: `{cfg.sae_release}` / `{cfg.sae_id_template.format(layer=args.layer)}`. "
         "Top-activating token (in [[ ]]) per corpus line. Labels are guesses from these contexts.\n"]
for fid in feats:
    hs = sorted(hits[fid], reverse=True)[:args.top]
    lines.append(f"## Feature {fid}  ({len(hits[fid])} of {len(corpus)} lines active)\n")
    for a, ctx in hs:
        lines.append(f"- {a:.1f}: `{ctx.strip()}`")
    lines.append("\nLabel: _\n")
out = Path("docs") / f"features_L{args.layer}.md"
out.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\nWrote {out}")
