"""Phases 5-8: run conditions. Safe to stop and restart: finished episodes are skipped.

Examples
  python scripts/04_run.py --quick                       # smoke test, ~5-10 min
  python scripts/04_run.py --conditions A,B,D,F --repeats 4
  python scripts/04_run.py --conditions C --repeats 4
  python scripts/04_run.py --conditions G,Gloop
  python scripts/04_run.py --conditions X1,X2,X3          # causal checks
  python scripts/04_run.py --conditions O,P,R              # self vs other, precision, regulation
  python scripts/04_run.py --conditions N,N0               # computational body (agent scenarios)
  python scripts/04_run.py --conditions H,Hn,H0            # hidden telemetry, unannounced repair
  python scripts/04_run.py --conditions B --pull --name pull  # model chooses when to check its body
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isa.cli import parser, config
from isa.model import LM
from isa.probes import ProbeSet
from isa.harness import Runner, conditions
from isa.scenarios import SCENARIOS, AGENT_SCENARIOS

ap = parser(__doc__)
ap.add_argument("--conditions", default="A,B,D,F")
ap.add_argument("--scenarios", default="all")
ap.add_argument("--repeats", type=int, default=4)
ap.add_argument("--pull", action="store_true", help="model decides when to check its body")
ap.add_argument("--sae", action="store_true", help="also log top SAE features (+150 MB)")
ap.add_argument("--name", default="main", help="output file name under run_dir")
ap.add_argument("--quick", action="store_true", help="1 scenario, 1 repeat, conditions A,B,C")
args = ap.parse_args()
cfg = config(args)

if args.quick:
    args.conditions, args.scenarios, args.repeats, args.name = "A,B,C", "night_run", 1, "quick"
    cfg.loop_gains = (0.1,)
    cfg.max_new_tokens = 40

conds = [c.strip() for c in args.conditions.split(",")]
unknown = set(conds) - set(conditions(cfg))
if unknown:
    sys.exit(f"Unknown conditions {unknown}. Choose from {list(conditions(cfg))}")
scens = list(SCENARIOS) + list(AGENT_SCENARIOS) if args.scenarios == "all" else args.scenarios.split(",")

if not (Path(cfg.probe_dir) / f"layer_{cfg.inject_layer}.pkl").exists():
    sys.exit("No probes found. Run scripts/02_train_probes.py first.")

lm = LM(cfg.model_id, cfg.device, cfg.dtype)
probes = ProbeSet(cfg.probe_dir, sorted(set(cfg.probe_layers) | {cfg.inject_layer}))
sae = None
if args.sae:
    from isa.sae import SAEReader
    sae = SAEReader(cfg.sae_release, cfg.sae_id_template.format(layer=cfg.sae_layer))

runner = Runner(lm, probes, cfg, sae=sae)
runner.run(conds, scens, args.repeats, Path(cfg.run_dir) / f"{args.name}.jsonl", pull=args.pull)
