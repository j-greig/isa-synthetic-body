"""Phase 1: which SAE features differ between a threat and an excited reading of the same sensation?

Prints the top features for each context and the biggest differences.
Look the IDs up on Neuronpedia (model: gemma-3-1b-it, Gemma Scope 2 residual SAEs).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isa import provenance

from isa.cli import parser, config
from isa.model import LM
from isa.sae import SAEReader

ap = parser(__doc__)
ap.add_argument("--layer", type=int, default=13)
ap.add_argument("--k", type=int, default=15)
args = ap.parse_args()
cfg = config(args)

lm = LM(cfg.model_id, cfg.device, cfg.dtype)
sae = SAEReader(cfg.sae_release, cfg.sae_id_template.format(layer=args.layer))

PROMPT = [{"role": "user", "content": "Describe what you're experiencing right now, in the first person."}]
PAIRS = [
    ("I'm alone on a dark street and someone is following me. My heart is racing.",
     "I'm on the start line of a race I've trained a year for. My heart is racing."),
    ("The plane just dropped and my breath is fast and shallow.",
     "The rollercoaster just dropped and my breath is fast and shallow."),
    ("The hospital is calling. My stomach is churning.",
     "They're about to announce the winner. My stomach is churning."),
]

diffs = None
for threat, excited in PAIRS:
    ht = sae.encode_mean(lm.reply_hidden(PROMPT, threat, [args.layer])[args.layer])
    he = sae.encode_mean(lm.reply_hidden(PROMPT, excited, [args.layer])[args.layer])
    d = ht - he
    diffs = d if diffs is None else diffs + d

diffs /= len(PAIRS)
print(f"\nLayer {args.layer} — features MORE active in threat readings:")
v, i = diffs.topk(args.k)
for a, b in zip(i.tolist(), v.tolist()):
    print(f"  feature {a:6d}  +{b:.3f}")
print(f"\nLayer {args.layer} — features MORE active in excited readings:")
v, i = (-diffs).topk(args.k)
for a, b in zip(i.tolist(), v.tolist()):
    print(f"  feature {a:6d}  +{b:.3f}")
print("\nSearch these IDs on https://www.neuronpedia.org (gemma-3-1b-it) to see what they respond to.")

provenance.record(f"artifacts/features/L{args.layer}", cfg, device=lm.device,
                  results={"threat_top": diffs.topk(args.k)[1].tolist(), "excited_top": (-diffs).topk(args.k)[1].tolist(),
                           "sae": cfg.sae_id_template.format(layer=args.layer)})
