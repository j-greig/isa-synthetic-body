"""Shared command-line options."""
from __future__ import annotations

import argparse

from isa.config import Config, from_args


def parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--model_id", help=f"default {Config.model_id}")
    ap.add_argument("--device", help="auto | mps | cuda | cpu")
    ap.add_argument("--dtype", help="auto | bfloat16 | float32")
    ap.add_argument("--probe_dir")
    ap.add_argument("--run_dir")
    ap.add_argument("--max_new_tokens", type=int)
    ap.add_argument("--inject_strength", type=float)
    return ap


def config(args) -> Config:
    return from_args(args)
