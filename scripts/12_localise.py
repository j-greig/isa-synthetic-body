"""Phase 2 experiment A: which sentence was perturbed? (isa/localise.py, docs/NEXT.md)

  python scripts/12_localise.py --vectors none,concept,random --framings self,other --strengths 0.5,1,2 --n 50
Strengths are multiples of the mean token norm at the injection layer (applied only to one sentence's tokens).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isa import provenance
from isa.cli import parser, config
from isa.model import LM
from isa.probes import ProbeSet
from isa.localise import Localiser, plan, run

ap = parser(__doc__)
ap.add_argument("--vectors", default="none,concept,random")
ap.add_argument("--framings", default="self,other")
ap.add_argument("--strengths", default="0.5,1,2")
ap.add_argument("--signs", default="1", help="for concept: '1' or '1,-1'")
ap.add_argument("--layer", type=int, help="injection layer (default cfg.inject_layer)")
ap.add_argument("--n", type=int, default=50, help="trials per cell")
ap.add_argument("--name", default="localise")
ap.add_argument("--concept_from", help="use an appraisal direction instead of arousal: artifacts/<run>/directions.npz:<appraisal>:<layer>")
args = ap.parse_args()
cfg = config(args)
layer = args.layer if args.layer is not None else cfg.inject_layer

lm = LM(cfg.model_id, cfg.device, cfg.dtype)
probes = ProbeSet(cfg.probe_dir, [cfg.inject_layer])
concept, norm = probes.direction(cfg.inject_layer, "arousal")
if args.concept_from:
    import numpy as np, torch
    from isa.appraisal_steer import APPRAISALS
    f, app, L = args.concept_from.split(":")
    z = np.load(f)
    concept, norm = torch.from_numpy(z["dirs"][int(L), APPRAISALS.index(app)]), float(z["norms"][int(L)])
    print(f"concept = {app} direction at layer {L} from {f}; norm {norm:.1f}")
if layer != cfg.inject_layer:
    print(f"note: concept direction and norm come from layer {cfg.inject_layer}, injecting at {layer}")
loc = Localiser(lm, concept, norm, layer)
trials = plan(args.vectors.split(","), args.framings.split(","), [float(s) for s in args.strengths.split(",")],
              layer, args.n, signs=tuple(int(s) for s in args.signs.split(",")))
out = Path(cfg.run_dir) / f"{args.name}.jsonl"
run(loc, trials, out)
provenance.record(out, cfg, device=lm.device,
                  data={f"probes_L{cfg.inject_layer}": Path(cfg.probe_dir) / f"layer_{cfg.inject_layer}.pkl"},
                  results={"vectors": args.vectors, "framings": args.framings, "strengths": args.strengths,
                           "signs": args.signs, "layer": layer, "n_per_cell": args.n, "token_norm": norm,
                           "concept_from": args.concept_from or "arousal probe"})
