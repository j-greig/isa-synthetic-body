"""Phase 2 C0: appraisal steering replication (Tak et al. 2025) on Gemma 3. See isa/appraisal_steer.py.

  python scripts/14_appraisal_steer.py --model_id <1b> --name ap_steer_1b --probe_per_emotion 150 --eval_per_emotion 15 \
      --centres 7,10,13,16,19 --betas 0.1,0.25,0.5 --n_random 20
Step 1 (cached): clean pass on probe items -> activations, labels. Step 2: Ridge directions per layer.
Step 3: steering on held-out items. Outputs in artifacts/<name>/ and runs/<name>.jsonl.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from isa import provenance
from isa.cli import parser, config
from isa.model import LM
from isa.appraisal_steer import (Steerer, load_corpus, fit_directions, run_steering, LABELS, APPRAISALS)

ap = parser(__doc__)
ap.add_argument("--corpus", default="data/crowd-enVent/corpus/crowd-enVent_generation.tsv")
ap.add_argument("--name", default="ap_steer")
ap.add_argument("--probe_per_emotion", type=int, default=150)
ap.add_argument("--eval_per_emotion", type=int, default=15)
ap.add_argument("--centres", default="7,10,13,16,19")
ap.add_argument("--betas", default="0.1,0.25,0.5")
ap.add_argument("--appraisals", default="pleasantness,other_responsblt,self_responsblt,self_control,predict_event")
ap.add_argument("--n_random", type=int, default=20)
ap.add_argument("--max_items", type=int, help="debug: cap probe and eval items")
args = ap.parse_args()
cfg = config(args)

art = Path("artifacts") / args.name
art.mkdir(parents=True, exist_ok=True)
d = load_corpus(args.corpus, seed=0)
ev = d.groupby("label", group_keys=False).head(args.eval_per_emotion)
pr = d.drop(ev.index).groupby("label", group_keys=False).head(args.probe_per_emotion)
if args.max_items:
    ev, pr = ev.head(args.max_items), pr.head(args.max_items)
print(f"probe items {len(pr)}, eval items {len(ev)}")

lm = LM(cfg.model_id, cfg.device, cfg.dtype)
st = Steerer(lm)

# Step 1: clean pass (cached)
cache = art / "clean.npz"
if cache.exists():
    z = np.load(cache)
    H, LP = z["H"], z["LP"]
    print("loaded cached activations", H.shape)
else:
    Hs, LPs = [], []
    for i, t in enumerate(pr["text"]):
        lp, h = st.clean(t)
        LPs.append(lp); Hs.append(h.astype(np.float16))
        if (i + 1) % 100 == 0:
            print(f"clean {i + 1}/{len(pr)}", flush=True)
    H, LP = np.stack(Hs), np.stack(LPs)
    np.savez(cache, H=H, LP=LP, text_id=pr["text_id"].values)
pred = np.array(LABELS)[LP.argmax(1)]
correct = pred == pr["label"].values
acc = {lab: float(correct[pr["label"].values == lab].mean()) for lab in LABELS}
print(f"clean accuracy {correct.mean():.3f}", acc)

# Step 2: directions on correctly labelled items (Tak et al.)
Hc = H[correct].astype(np.float32)
Y = pr.loc[correct, APPRAISALS].values.astype(float)
dirs, r = fit_directions(Hc, Y)
norms = np.linalg.norm(H.astype(np.float32), axis=2).mean(0)
np.savez(art / "directions.npz", dirs=dirs, cv_r=r, norms=norms)
best = {a: int(r[:, j].argmax()) for j, a in enumerate(APPRAISALS)}
(art / "probe_report.json").write_text(json.dumps({
    "n_probe": int(len(pr)), "n_correct": int(correct.sum()), "clean_accuracy": float(correct.mean()), "per_label_accuracy": acc,
    "cv_r_by_layer": {a: [round(float(x), 3) for x in r[:, j]] for j, a in enumerate(APPRAISALS)},
    "best_layer": best, "mean_last_token_norm": [round(float(x), 1) for x in norms]}, indent=1))
print("CV r (best layer):", {a: round(float(r[best[a], j]), 2) for j, a in enumerate(APPRAISALS)})

# Step 3: steering on held-out items
out = Path(cfg.run_dir) / f"{args.name}.jsonl"
combos = [(("pleasantness", 1), ("other_responsblt", 1)), (("pleasantness", 1), ("other_responsblt", -1)),
          (("pleasantness", -1), ("other_responsblt", 1)), (("pleasantness", -1), ("other_responsblt", -1))]
run_steering(st, list(ev["text"]), [int(x) for x in ev["text_id"]], dirs, norms,
             [int(c) for c in args.centres.split(",")], [float(b) for b in args.betas.split(",")],
             args.appraisals.split(","), args.n_random, out, combos=combos)
(art / "eval_items.json").write_text(json.dumps({int(t): l for t, l in zip(ev["text_id"], ev["label"])}))
provenance.record(out, cfg, device=lm.device, data={"corpus": Path(args.corpus)},
                  results={"probe_items": int(len(pr)), "eval_items": int(len(ev)), "clean_accuracy": float(correct.mean()),
                           "centres": args.centres, "betas": args.betas, "appraisals": args.appraisals,
                           "n_random": args.n_random, "artifacts": str(art)})
