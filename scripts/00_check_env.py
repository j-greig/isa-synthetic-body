"""Phase 0: check the machine can load the model and generate."""
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from isa.cli import parser, config
from isa.model import LM, pick_device

args = parser(__doc__).parse_args()
cfg = config(args)

print(f"Python {platform.python_version()} | torch {torch.__version__} | {platform.platform()}")
if platform.system() == "Darwin":
    for key in ("machdep.cpu.brand_string", "hw.memsize"):
        try:
            v = subprocess.check_output(["sysctl", "-n", key]).decode().strip()
            print(f"{key}: {int(v) / 2**30:.0f} GB" if key == "hw.memsize" else f"{key}: {v}")
        except Exception:
            pass
print(f"MPS available: {torch.backends.mps.is_available()} | CUDA: {torch.cuda.is_available()}")
print(f"Device chosen: {pick_device(cfg.device)}")

t = time.time()
lm = LM(cfg.model_id, cfg.device, cfg.dtype)
print(f"Loaded {cfg.model_id} ({lm.n_layers} layers, {lm.dtype}) in {time.time() - t:.0f}s")

msgs = [{"role": "user", "content": "My heart is racing. In one sentence, what might that mean?"}]
t = time.time()
text, ent = lm.generate(msgs, max_new_tokens=40)
dt = time.time() - t
print(f"\nReply: {text}\n\n~{40 / dt:.1f} tokens/sec | mean entropy {ent:.2f}")
print("\nOK. Now check Activity Monitor > Memory: pressure should be green or yellow.")
