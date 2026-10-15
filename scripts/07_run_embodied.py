"""Beyond tokens: neutral arithmetic task with hidden, meaning-free and native body channels.

  python scripts/07_run_embodied.py --conditions T0,TR,TC,TS,TL,TU,TN,TNC,TN0 --repeats 2
  python scripts/07_run_embodied.py --quick
See isa/embodied.py for the conditions. Hidden states go to runs/<name>_hidden/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isa import provenance

from isa.cli import parser, config
from isa.model import LM
from isa.probes import ProbeSet
from isa import appraisal
from isa.embodied import EmbodiedRunner, tconditions, TASKS

ap = parser(__doc__)
ap.add_argument("--conditions", default="T0,TR,TC,TS,TL,TU,TN,TNC,TN0,TRx")
ap.add_argument("--tasks", default="add,mul")
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--name", default="embodied")
ap.add_argument("--rep_start", type=int, default=0, help="first repeat index (new seeds for replications)")
ap.add_argument("--strength", type=float, help="injection scale (default cfg.embodied_strength)")
ap.add_argument("--quick", action="store_true", help="T0,TR,TS,TN; add only; 1 repeat")
args = ap.parse_args()
cfg = config(args)
if args.quick:
    args.conditions, args.tasks, args.repeats, args.name = "T0,TR,TS,TN", "add", 1, "embodied_quick"

if args.strength is not None:
    cfg.embodied_strength = args.strength
names = [c.strip() for c in args.conditions.split(",")]
bad = set(names) - set(tconditions())
if bad:
    sys.exit(f"Unknown conditions {bad}. Choose from {list(tconditions())}")
tasks = [t for t in args.tasks.split(",") if t in TASKS]

lm = LM(cfg.model_id, cfg.device, cfg.dtype)
probes = ProbeSet(cfg.probe_dir, sorted(set(cfg.probe_layers) | {cfg.inject_layer}))
ap_set = appraisal.maybe_load("artifacts/appraisal", cfg.probe_layers)
runner = EmbodiedRunner(lm, probes, cfg, ap_set)
out = Path(cfg.run_dir) / f"{args.name}.jsonl"
runner.run(names, tasks, args.repeats, out, rep_start=args.rep_start)

provenance.record(out, cfg, device=lm.device,
                  data={f"probes_L{L}": Path(cfg.probe_dir) / f"layer_{L}.pkl" for L in probes.layers},
                  results={"conditions": names, "tasks": tasks, "repeats": args.repeats, "rep_start": args.rep_start, "strength": cfg.embodied_strength,
                           "hidden_dir": str(out.parent / (out.stem + "_hidden"))})
