"""Phase 2: train threat-vs-challenge and arousal probes; save steering directions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isa import provenance

from isa.cli import parser, config
from isa.model import LM
from isa import probes

ap = parser(__doc__)
ap.add_argument("--per_ctx", type=int, default=7, help="passages per context (15 contexts x 3 classes)")
args = ap.parse_args()
cfg = config(args)

lm = LM(cfg.model_id, cfg.device, cfg.dtype)
layers = sorted(set(cfg.probe_layers) | {cfg.inject_layer})
rep = probes.train(lm, layers, cfg.probe_dir, per_ctx=args.per_ctx)

weak = [L for L, r in rep.items() if L in cfg.probe_layers and r["appraisal_test_acc"] < 0.85]
if weak:
    print(f"\nWarning: appraisal accuracy below 0.85 at layers {weak}. "
          "Try other layers (--probe layers in isa/config.py) or more passages (--per_ctx).")
print(f"\nSaved to {cfg.probe_dir}/")

provenance.record(cfg.probe_dir, cfg, device=lm.device, results=rep)
