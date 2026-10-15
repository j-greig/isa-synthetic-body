"""The simulated body as an MCP server, so any MCP-capable agent can "have" it.

Run:  python -m isa.mcp_server            (stdio, for Claude Desktop etc.)
      python -m isa.mcp_server --http 8765

The experiment harness uses the same Body class in-process (faster, and the
small model can't drive MCP itself). This server is for demos and for trying
the body on larger agents.
"""
from __future__ import annotations

import argparse

from fastmcp import FastMCP

from isa.body import Body, BodyParams, CHANNELS

mcp = FastMCP("synthetic-body")
_body = Body(BodyParams())


@mcp.tool
def get_heartrate() -> str:
    """Read your heart rate (bpm) and its trend."""
    return _body.describe("heart")


@mcp.tool
def get_breath() -> str:
    """Read your breathing rate and quality."""
    return _body.describe("breath")


@mcp.tool
def get_gut() -> str:
    """Read your gut tension (0-10)."""
    return _body.describe("gut")


@mcp.tool
def get_fatigue() -> str:
    """Read your energy level (0-10)."""
    return _body.describe("energy")


@mcp.tool
def get_all() -> str:
    """Read every body channel at once."""
    return _body.report(CHANNELS)


@mcp.tool
def perturb(event: float = 0.5, threat: float = 0.0) -> dict:
    """Advance the body one step. `event` (0-1) is how stimulating the moment is;
    `threat` (-1..1) is how threatening you judge it, which feeds back when loop gain > 0."""
    return _body.step(event=event, threat=threat)


@mcp.tool
def configure(gain: float | None = None, decay: float | None = None, reset: bool = False) -> dict:
    """Set loop gain / recovery rate, or reset the body to baseline."""
    global _body
    if reset:
        _body = Body(BodyParams(**{**_body.params.__dict__}))
    if gain is not None:
        _body.params.gain = gain
    if decay is not None:
        _body.params.decay = decay
    return _body.to_dict()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", type=int, default=None, help="serve over HTTP on this port")
    args = ap.parse_args()
    if args.http:
        mcp.run(transport="http", port=args.http)
    else:
        mcp.run()
