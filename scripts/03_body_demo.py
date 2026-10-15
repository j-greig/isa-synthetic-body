"""Phase 3: plot the body on its own (no model). Shows scripted curve, open loop,
and closed loop with constant threat vs. constant effort appraisal."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isa import provenance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isa.body import Body, BodyParams, SCRIPTED_A, EVENTS

out = Path("artifacts/figures")
out.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
b = Body(seed=0)
rows = [b.step(set_a=a) for a in SCRIPTED_A]
ax = axes[0]
ax.plot([r["a"] for r in rows], label="arousal a (scripted)")
ax.plot([r["a_breath"] for r in rows], label="breath (medium)")
ax.plot([r["a_gut"] for r in rows], label="gut (slow)")
ax.plot([r["fatigue"] for r in rows], label="fatigue")
ax.set_title("Scripted curve: channels at different speeds")
ax.set_xlabel("turn"); ax.legend(fontsize=8)

ax = axes[1]
for gain in (0.0, 0.1, 0.2):
    for threat, ls in ((1.0, "-"), (-1.0, "--")):
        b = Body(BodyParams(gain=gain, noise=0), seed=0)
        a = [b.step(event=EVENTS[t], threat=threat)["a"] for t in range(len(EVENTS))]
        ax.plot(a, ls, label=f"g={gain} {'threat' if threat > 0 else 'effort'}")
ax.set_title("Closed loop: constant appraisal")
ax.set_xlabel("turn"); ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(out / "body_demo.png", dpi=130)
print(f"Saved {out / 'body_demo.png'}")

provenance.record(out / "body_demo.png")
