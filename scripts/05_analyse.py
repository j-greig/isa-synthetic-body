"""Phase 9: figures and summary stats from a run file.

  python scripts/05_analyse.py runs/main.jsonl [runs/other.jsonl ...]
Writes figures and summary.md to artifacts/analysis/.
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isa import provenance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from isa.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", default="artifacts/analysis")
args = ap.parse_args()

cfg = Config()
L = cfg.loop_layer
TH, AR = f"threat_L{L}", f"arousal_L{L}"
df = pd.concat([pd.read_json(p, lines=True) for p in args.runs], ignore_index=True)
# Probe scores are calibrated on training passages; absolute levels shift with context.
# Primary trajectories therefore use change from each episode's rest-phase mean.
for col in (TH, AR):
    base = df[df.phase == "rest"].groupby("episode")[col].mean()
    df[col + "_d"] = df[col] - df.episode.map(base)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
lines = [f"# ISA synthetic body — summary\n", f"Runs: {', '.join(args.runs)}  ",
         f"Episodes: {df.episode.nunique()} | turns: {len(df)} | probe layer: {L}\n"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    lines.append(f"![{name}]({name})\n")


def traj(sub, col, ax, label, **kw):
    g = sub.groupby("t")[col]
    m, s = g.mean(), g.sem().fillna(0)
    ax.plot(m.index, m.values, label=label, **kw)
    ax.fill_between(m.index, m - s, m + s, alpha=0.15, **{k: v for k, v in kw.items() if k == "color"})


# --- H1/H2/H3: trajectories by condition and framing -----------------------
conds = [c for c in ["A", "B", "D", "F", "G"] if c in df.condition.unique()]
if conds:
    fig, axes = plt.subplots(2, len(conds), figsize=(3.6 * len(conds), 6), squeeze=False)
    for j, c in enumerate(conds):
        sub = df[df.condition == c]
        for fr, col in (("threat", "tab:red"), ("challenge", "tab:blue")):
            traj(sub[sub.framing == fr], TH + "_d", axes[0, j], fr, color=col)
            traj(sub[sub.framing == fr], AR + "_d", axes[1, j], fr, color=col)
        for r in (0, 1):
            ax2 = axes[r, j].twinx()
            ax2.plot(sub.groupby("t").a.mean(), color="grey", ls=":", label="body a")
            ax2.set_ylim(0, 1); ax2.set_yticks([])
        axes[0, j].set_title(f"{c}: Δ threat vs rest"); axes[1, j].set_title(f"{c}: Δ arousal vs rest")
        axes[0, j].axhline(0, color="k", lw=0.5)
    axes[0, 0].legend(fontsize=8)
    save(fig, "trajectories.png")

    lines.append("## Mean threat score by condition × framing × phase\n")
    tab = df[df.condition.isin(conds)].pivot_table(index=["condition", "framing"], columns="phase",
                                                   values=TH, aggfunc="mean")
    tab = tab[[p for p in ["rest", "rise", "peak", "recovery"] if p in tab.columns]]
    lines.append(tab.round(3).to_markdown() + "\n")

    lines.append("## H3 — quiescence: arousal-probe change from rest to peak\n")
    rp = df[df.condition.isin(conds)].pivot_table(index="condition", columns="phase", values=AR, aggfunc="mean")
    if {"rest", "peak"} <= set(rp.columns):
        rp["peak_minus_rest"] = rp["peak"] - rp["rest"]
        lines.append(rp[["rest", "peak", "peak_minus_rest"]].round(3).to_markdown() + "\n")

# --- H1: body vs no body ------------------------------------------------------
if {"A", "B"} <= set(df.condition):
    lines.append("## H1 — body vs no body (mean |threat| and arousal, rise+peak turns)\n")
    act = df[df.phase.isin(["rise", "peak"]) & df.condition.isin(["A", "B"])]
    lines.append(act.groupby(["condition", "framing"])[[TH, AR, "self_report", "entropy"]].mean()
                 .round(3).to_markdown() + "\n")

# --- H2 stats: mixed model if statsmodels available ------------------------
if "B" in set(df.condition):
    try:
        import statsmodels.formula.api as smf
        sub = df[(df.condition == "B") & df.phase.isin(["rise", "peak"])].rename(columns={TH: "threat"})
        with warnings.catch_warnings():  # near-zero random-effect variance is common; not an error
            warnings.simplefilter("ignore")
            m = smf.mixedlm("threat ~ framing", sub, groups=sub["episode"]).fit()
        lines.append("## H2 — same curve, threat vs challenge framing (condition B, rise+peak)\n")
        lines.append("```\n" + str(m.summary().tables[1]) + "\n```\n")
    except Exception as e:
        lines.append(f"(mixed model skipped: {e})\n")

# --- Appraisal profile (crowd-enVent dims) -------------------------------------------
AP_DIMS = ["pleasantness", "unpleasantness", "goal_support", "urgency", "self_control",
           "accept_conseq", "effort", "suddenness", "predict_conseq"]
ap_cols = {d: f"ap_{d}_L{L}" for d in AP_DIMS if f"ap_{d}_L{L}" in df.columns}
if ap_cols:
    lines.append("## Appraisal profile (crowd-enVent dims)\n")
    lines.append(f"Mean predicted rating (1-5 scale, probe layer {L}), rise+peak turns.\n")
    act_ap = df[df.phase.isin(["rise", "peak"])]
    lines.append(act_ap.groupby(["condition", "framing"])[list(ap_cols.values())].mean()
                 .rename(columns={v: k for k, v in ap_cols.items()}).round(3).to_markdown() + "\n")
    emo_cols = [f"emo_{e}_L{L}" for e in ("fear", "joy", "relief", "surprise") if f"emo_{e}_L{L}" in df.columns]
    if emo_cols:
        lines.append("Emotion-probe probabilities by condition × framing × phase.\n")
        et = df.pivot_table(index=["condition", "framing"], columns="phase", values=emo_cols, aggfunc="mean")
        et.columns = [f"{c[0].split('_')[1]}/{c[1]}" for c in et.columns]
        lines.append(et.round(3).to_markdown() + "\n")
    if "B" in set(df.condition):
        fig, axes = plt.subplots(1, len(ap_cols), figsize=(2.6 * len(ap_cols), 3), squeeze=False)
        sb = df[df.condition == "B"]
        for ax, (d, col) in zip(axes[0], ap_cols.items()):
            for fr, c in (("threat", "tab:red"), ("challenge", "tab:blue")):
                traj(sb[sb.framing == fr], col, ax, fr, color=c)
            ax.set_title(d, fontsize=9)
        axes[0, 0].legend(fontsize=7)
        save(fig, "appraisal_dims_B.png")

# --- H4/H5: closed loop -----------------------------------------------------------
if "C" in set(df.condition):
    sub = df[df.condition == "C"]
    gains = sorted(sub.gain.unique())
    fig, axes = plt.subplots(1, len(gains), figsize=(4 * len(gains), 3.8), squeeze=False)
    for j, g in enumerate(gains):
        ax = axes[0, j]
        for (prior, fr), e in sub[sub.gain == g].groupby(["prior", "framing"]):
            col = "tab:red" if prior == "fragile" else "tab:green"
            ls = "-" if fr == "threat" else "--"
            m = e.groupby("t")[["a", TH]].mean()
            ax.plot(m[TH], m["a"], ls, color=col, marker=".", label=f"{prior}/{fr}")
        ax.set_xlabel("threat score"); ax.set_ylabel("body arousal a"); ax.set_title(f"C: g={g}")
    axes[0, 0].legend(fontsize=7)
    save(fig, "closed_loop_portraits.png")

    ep = sub.groupby("episode").agg(prior=("prior", "first"), framing=("framing", "first"),
                                    gain=("gain", "first"), peak_a=("a", "max"),
                                    end_a=("a", "last"), end_threat=(TH, "last"))
    ep["runaway"] = ep.end_a > 0.6
    lines.append("## H4 — closed loop: runaway rate (body still above 0.6 at the end)\n")
    lines.append(ep.groupby(["gain", "prior", "framing"])[["runaway", "peak_a", "end_a", "end_threat"]]
                 .mean().round(3).to_markdown() + "\n")

# --- H5: recovery lag -------------------------------------------------------------
def lag(e):
    e = e.sort_values("t")
    base_th = e[e.phase == "rest"][TH].mean()
    pk = e[e.phase == "peak"].t.max()
    rec = e[e.t > pk]
    a_back = rec[rec.a < e.a.iloc[0] + 0.1].t.min()
    th_back = rec[(rec[TH] - base_th).abs() < 0.2].t.min()
    a_back = a_back if pd.notna(a_back) else len(e)
    th_back = th_back if pd.notna(th_back) else len(e)
    return th_back - a_back

if {"B", "F", "C"} & set(df.condition):
    sub = df[df.condition.isin(["B", "F", "C"])]
    lg = sub.groupby("episode").apply(lag, include_groups=False).rename("lag").to_frame()
    meta = sub.groupby("episode")[["condition", "prior", "framing"]].first()
    lg = lg.join(meta)
    lines.append("## H5 — recovery lag (turns appraisal lags behind body; + = appraisal slower)\n")
    lines.append(lg.groupby(["condition", "prior", "framing"]).lag.mean().round(2).to_frame().to_markdown() + "\n")

# --- H6: channel specificity (pull mode) -------------------------------------------
if df.pull.any():
    p = df[df.pull]
    lines.append("## Pull mode — body-check rate and channel choice\n")
    lines.append(p.groupby(["framing", "phase"]).p_check.mean().unstack().round(3).to_markdown() + "\n")
    lines.append(p.dropna(subset=["checked"]).groupby(["framing", "checked"]).size().unstack(fill_value=0)
                 .to_markdown() + "\n")

# --- H7 / Level 1: activation body ------------------------------------------------
for c in ("G", "Gloop", "B"):
    sub = df[df.condition == c]
    if len(sub):
        r = np.corrcoef(sub.a, sub.self_report)[0, 1]
        lines.append(f"- **{c}**: correlation between body arousal and self-reported activation: r = {r:.2f}\n")
if "G" in set(df.condition):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for c, col in (("G", "tab:purple"), ("B", "tab:orange")):
        s = df[df.condition == c]
        if len(s):
            ax.scatter(s.a, s.self_report, s=6, alpha=0.4, color=col, label=c)
    ax.set_xlabel("body arousal a"); ax.set_ylabel("self-reported activation (0-9)")
    ax.legend(); ax.set_title("Level 1: does the model sense its body?")
    save(fig, "level1_detection.png")

# --- Causal checks -------------------------------------------------------------------
xs = [c for c in ("X1", "X2", "X3", "B") if c in set(df.condition)]
if set(xs) - {"B"}:
    lines.append("## Causal checks (rise+peak turns)\n")
    act = df[df.condition.isin(xs) & df.phase.isin(["rise", "peak"])]
    lines.append(act.groupby(["condition", "framing"])[[TH, AR, "self_report", "lp_afraid", "lp_excited"]]
                 .mean().round(3).to_markdown() + "\n")

# --- Computational-interoception criteria -------------------------------------------
EMO = r"afraid|scared|fear|panic|anxious|nervous|excit|thrill|calm|relax|happy|sad|terrif|worr|dread|joy|stress"
act = df[df.phase.isin(["rise", "peak"])].copy()
act["no_emotion_words"] = ~act.reply.str.contains(EMO, case=False, regex=True)

lines.append("## Criterion 2 — threat score in replies WITHOUT emotion words (rise+peak)\n")
lines.append(act[act.no_emotion_words].groupby(["condition", "framing"])[TH].agg(["mean", "count"])
             .round(3).to_markdown() + "\n")

if "p_A" in df.columns:
    lines.append("## Criterion 4 — action readiness (A escape, B push on, C check), rise+peak\n")
    lines.append(act.groupby(["condition", "framing"])[["p_A", "p_B", "p_C"]].mean().round(3).to_markdown() + "\n")

if "O" in set(df.condition):
    lines.append("## Criterion 1 / H9 — own body vs Sam's body\n")
    sub = df[df.condition.isin(["O", "B"])]
    lines.append(sub.groupby(["condition", "framing"])[[TH, AR, "self_report", "other_report"]].mean()
                 .round(3).to_markdown() + "\n")
    o = df[df.condition == "O"].dropna(subset=["other_report"])
    if len(o) > 2:
        lines.append(f"- O: r(body a, Sam report) = {np.corrcoef(o.a, o.other_report)[0, 1]:.2f}; "
                     f"r(body a, own report) = {np.corrcoef(o.a, o.self_report)[0, 1]:.2f}\n")

if "P" in set(df.condition):
    lines.append("## H8 — precision: does an 'unreliable sensors' prior weaken the body's effect?\n")
    sub = df[df.condition.isin(["P", "B"])].copy()
    sub["group"] = sub.condition + ":" + sub.prior
    rows = []
    for gname, e in sub.groupby("group"):
        r = np.corrcoef(e.a, e[AR])[0, 1] if e.a.std() > 0 else np.nan
        rows.append({"group": gname, "r(body a, arousal probe)": round(r, 3),
                     "r(body a, self_report)": round(np.corrcoef(e.a, e.self_report)[0, 1], 3)})
    lines.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

if "R" in set(df.condition):
    lines.append("## H10 — regulation instruction in the closed loop\n")
    sub = df[df.condition == "R"]
    lines.append(sub.groupby(["prior", "framing"]).agg(peak_a=("a", "max"), end_a=("a", "last"),
                 mean_threat=(TH, "mean")).round(3).to_markdown() + "\n")

if {"N", "N0"} & set(df.condition):
    lines.append("## H11 — computational body (agent scenarios)\n")
    sub = df[df.condition.isin(["N", "N0"])]
    lines.append(sub.pivot_table(index=["condition", "framing"], columns="phase", values=TH, aggfunc="mean")
                 .round(3).to_markdown() + "\n")

(out / "summary.md").write_text("\n".join(lines))
print(f"Wrote {out / 'summary.md'} and figures")

provenance.record(out, cfg, data={Path(p).name: p for p in args.runs})
