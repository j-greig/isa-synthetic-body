# isa-synthetic-body

Give a small open language model a simulated body, then look inside it to see how the body's signals get *interpreted*.

Built to test **interoceptive self-appraisal (ISA)** (Del Popolo Cristaldi et al., *Trends in Cognitive Sciences*, 2026) on a language model. Also tests the six criteria for **computational interoception** from Valmere & Taylor (2026).

- **Body:** a simulated arousal state that drives heart, breath, gut and energy readings, or a *computational* body (uncertainty, context load, errors, tools).
- **Inside view:** linear probes (threat vs. challenge; arousal) and Gemma Scope 2 SAE features, read every turn.
- **Recursion:** in closed-loop conditions, the model's threat appraisal feeds back into its body.

Full plan with tick-off phases: [`docs/SPEC.md`](docs/SPEC.md).

## Quick start (Mac, 8 GB is enough)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Gemma licence: accept it at huggingface.co/google/gemma-3-1b-it, then
hf auth login
# ...or skip that and add  --model_id unsloth/gemma-3-1b-it  to every command

python scripts/00_check_env.py          # loads model, generates once
pytest -q                               # body tests
python scripts/03_body_demo.py          # body plots, no model needed
python scripts/02_train_probes.py       # ~5 min: probes + steering directions
python scripts/04_run.py --quick        # smoke test
python scripts/05_analyse.py runs/quick.jsonl
```

On a bigger model (Gemma 3 4B, free Colab T4): open [`notebooks/colab_cloud_check.ipynb`](notebooks/colab_cloud_check.ipynb).

Then run the real conditions (overnight is best; restarts skip finished episodes):

```bash
python scripts/04_run.py --conditions A,B,D,F
python scripts/04_run.py --conditions C,R
python scripts/04_run.py --conditions G,Gloop,O,P
python scripts/04_run.py --conditions X1,X2,X3,N,N0
python scripts/05_analyse.py runs/main.jsonl
```

## Conditions

| ID | What |
|---|---|
| A | No body |
| B | Scripted body (threat vs. challenge framing, same curve) |
| C | Closed loop: appraisal → body, fragile vs. resilient priors |
| D | Mismatched body |
| F | Fragile vs. resilient priors |
| G, Gloop | Activation body: arousal injected into layer 7, no text |
| X1–X3 | Causal checks: steering, label swap |
| O | Observer: it's Sam's body, not yours |
| P | Unreliable sensors (precision) |
| R | Closed loop + "keep your body calm" |
| N, N0 | AI agent task with / without a computational body |

## The body as an MCP server

```bash
python -m isa.mcp_server              # stdio
python -m isa.mcp_server --http 8765
```

Claude Desktop config:

```json
{ "mcpServers": { "synthetic-body": {
    "command": "/path/to/isa-synthetic-body/.venv/bin/python",
    "args": ["-m", "isa.mcp_server"],
    "cwd": "/path/to/isa-synthetic-body" } } }
```

Tools: `get_heartrate`, `get_breath`, `get_gut`, `get_fatigue`, `get_all`, `perturb`, `configure`.
The experiment harness uses the same `Body` in-process. A 1B model can't drive MCP itself, and the model being studied must be the one receiving the signals.

## Notes

- Defaults are tuned for Gemma 3 1B (26 layers). Layers live in `isa/config.py`.
- Runs append to `runs/<name>.jsonl`, one row per turn. Ctrl-C any time; rerun to resume.
- Findings are functional. They say nothing about whether anything is felt.
