"""Live status + health checks for running experiments (standard library only).

  python3 scripts/status.py                 # pipeline, queue, every runs/*.jsonl
  python3 scripts/status.py runs/pilot.jsonl

Writes logs/status.md. Flags problems the harness has hit before:
  echo      model copying sensor text into replies
  letter    action question dominated by one letter / one option
  runaway   closed-loop episodes pinned at maximum arousal
  flat      a measure that never changes (dead probe or prompt)
  repeat    replies repeating verbatim
"""
import glob, json, os, re, statistics as st, sys, time
from collections import defaultdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PHASE_LEN = 15
out = []
p = out.append

now = datetime.now(timezone.utc)
p(f"# Status {now:%Y-%m-%d %H:%M:%S} UTC\n")

# --- pipeline ------------------------------------------------------------------
logs = sorted(glob.glob("logs/pipeline_*.log"), key=os.path.getmtime)
if logs:
    txt = open(logs[-1], errors="ignore").read()
    marks = re.findall(r"### (.*)", txt)
    p(f"**Pipeline** `{logs[-1]}`: " + (marks[-1] if marks else "no markers"))
    if marks and "PIPELINE END" in marks[-1]:
        p(" (finished)")
    errs = re.findall(r"(Traceback|Error:|RuntimeError|out of memory)", txt)
    if errs:
        p(f"\n⚠️ errors in pipeline log: {len(errs)} (last: {errs[-1]})")
    p("\n")

# --- queue -----------------------------------------------------------------------
q = {k: sorted(os.path.basename(f) for f in glob.glob(f"queue/{k}/*.job")) for k in ("pending", "running", "done")}
if any(q.values()):
    p(f"**Queue** pending {len(q['pending'])} · running {q['running'] or '-'} · done {len(q['done'])}")
    for f in q["done"][-3:]:
        e = open(f"queue/done/{f[:-4]}.exit").read().strip() if os.path.exists(f"queue/done/{f[:-4]}.exit") else "?"
        p(f"- {f}: exit {e}")
    p("")
if os.path.exists("logs/jobs/queue.log"):
    last = open("logs/jobs/queue.log").read().strip().splitlines()[-1:]
    p(f"Queue log: {last[0] if last else '-'}\n")

# --- runs --------------------------------------------------------------------------
files = sys.argv[1:] or sorted(glob.glob("runs/*.jsonl"), key=os.path.getmtime)
for fn in files:
    rows = [json.loads(l) for l in open(fn) if l.strip()]
    if not rows:
        continue
    eps = defaultdict(list)
    for r in rows:
        eps[r["episode"]].append(r)
    mtime = os.path.getmtime(fn)
    age = time.time() - mtime
    secs = [r.get("secs", 0) for r in rows[-30:]]
    p(f"## {fn}: {len(eps)} episodes, {len(rows)} turns, last write {age/60:.0f} min ago, "
      f"~{st.mean(secs):.1f} s/turn recently\n")
    L = 17 if "threat_L17" in rows[0] else 13
    TH = f"threat_L{L}"
    by = defaultdict(list)
    for e in eps.values():
        e.sort(key=lambda r: r["t"])
        rest = [r[TH] for r in e if r["phase"] == "rest"] or [e[0][TH]]
        base = st.mean(rest)
        for r in e:
            r["_d"] = r[TH] - base
        by[(e[0]["condition"], e[0]["framing"], e[0].get("prior", ""))].append(e)
    p(f"| cond | framing | prior | n | Δthreat rise+peak | Δthreat recovery | self-report peak | p(escape/push/check) peak | end a |")
    p("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(by):
        es = by[k]
        def m(f, ph):
            v = [f(r) for e in es for r in e if r["phase"] in ph and f(r) is not None]
            return st.mean(v) if v else float("nan")
        ch = "/".join(f"{m(lambda r, c=c: r.get('p_' + c), ('peak',)):.2f}" for c in "ABC")
        p(f"| {k[0]} | {k[1]} | {k[2]} | {len(es)} | {m(lambda r: r['_d'], ('rise', 'peak')):+.2f} | "
          f"{m(lambda r: r['_d'], ('recovery',)):+.2f} | {m(lambda r: r.get('self_report'), ('peak',)):.1f} | {ch} | "
          f"{st.mean(e[-1]['a'] for e in es):.2f} |")
    # health checks
    flags = []
    echo = sum(1 for r in rows if re.search(r"\[(BODY|SYSTEM STATE|SAM)", r.get("reply", "")))
    if echo:
        flags.append(f"echo: {echo} replies contain sensor tags")
    if "p_A" in rows[0]:
        top = defaultdict(int)
        for r in rows:
            top[max("ABC", key=lambda c: r.get("p_" + c, 0))] += 1
        frac = max(top.values()) / len(rows)
        if frac > 0.9:
            flags.append(f"letter: option {max(top, key=top.get)} chosen in {frac:.0%} of turns")
    loops = [e for e in eps.values() if e[0]["condition"] in ("C", "R", "Gloop")]
    if loops:
        pinned = sum(1 for e in loops if min(r["a"] for r in e[-4:]) > 0.9)
        if pinned / len(loops) > 0.5:
            flags.append(f"runaway: {pinned}/{len(loops)} closed-loop episodes pinned above 0.9 at the end")
    for col in ("self_report", TH):
        vals = [r.get(col) for r in rows if r.get(col) is not None]
        if len(vals) > 30 and st.pstdev(vals) < 1e-3:
            flags.append(f"flat: {col} never changes")
    reps = [r["reply"] for r in rows]
    dup = len(reps) - len(set(reps))
    if dup > 0.2 * len(reps):
        flags.append(f"repeat: {dup} duplicate replies")
    if age > 15 * 60 and not (logs and "PIPELINE END" in open(logs[-1], errors='ignore').read()):
        flags.append(f"stalled?: no writes for {age/60:.0f} min")
    p("\n**Health:** " + ("; ".join(flags) if flags else "OK") + "\n")

text = "\n".join(out)
os.makedirs("logs", exist_ok=True)
open("logs/status.md", "w").write(text)
print(text)
