"""Phase 2c: train appraisal (21 dims) and emotion (13 classes) probes on crowd-enVent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isa import provenance

from isa.cli import parser, config
from isa.model import LM
from isa import appraisal

ap = parser(__doc__)
ap.add_argument("--corpus", default="data/crowd-enVent/corpus/crowd-enVent_generation.tsv")
ap.add_argument("--n_per_emotion", type=int, default=120)
args = ap.parse_args()
cfg = config(args)

lm = LM(cfg.model_id, cfg.device, cfg.dtype)
appraisal.train(lm, list(cfg.probe_layers), "artifacts/appraisal", args.corpus, args.n_per_emotion)
print("\nSaved to artifacts/appraisal/")

import json as _json
provenance.record("artifacts/appraisal", cfg, device=lm.device, data={"crowd_enVent": args.corpus},
                  results=_json.loads(Path("artifacts/appraisal/report.json").read_text()))
