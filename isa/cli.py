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
    ap.add_argument("--inject_layer", type=int, help="layer where hidden signals are added (Gemma 3 1B: 7; 4B: 9)")
    ap.add_argument("--probe_layers", help="comma-separated probe layers (1B: 13,17; 4B: 17,22)")
    ap.add_argument("--loop_layer", type=int)
    ap.add_argument("--embodied_strength", type=float)
    return ap


def config(args) -> Config:
    if getattr(args, "probe_layers", None) and isinstance(args.probe_layers, str):
        args.probe_layers = tuple(int(x) for x in args.probe_layers.split(","))
    return from_args(args)
