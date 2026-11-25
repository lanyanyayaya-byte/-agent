#!/usr/bin/env python3
"""Render a Mermaid mindmap (.mmd) into validated SVG and PNG via local mmdc."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _find_puppeteer_config() -> Path | None:
    env_cfg = os.environ.get("KNOWLEDGEAGENT_PUPPETEER_CONFIG")
    if env_cfg and Path(env_cfg).is_file():
        return Path(env_cfg)
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidate = Path(venv).parent / "cache" / "puppeteer-config.json"
        if candidate.is_file():
            return candidate
    return None


def _run_mmdc(
    mmdc: str,
    input_path: Path,
    output_path: Path,
    *,
    mermaid_config: Path | None,
    puppeteer_config: Path | None,
    scale: int | None = None,
) -> None:
    cmd = [mmdc, "-i", str(input_path), "-o", str(output_path), "-b", "white"]
    if mermaid_config and mermaid_config.is_file():
        cmd.extend(["-c", str(mermaid_config)])
    if puppeteer_config and puppeteer_config.is_file():
        cmd.extend(["-p", str(puppeteer_config)])
    if scale is not None:
        cmd.extend(["-s", str(scale)])

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        print(f"mmdc render failed for {output_path.name} (exit {proc.returncode}):\n{err}", file=sys.stderr)
        raise SystemExit(proc.returncode or 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Mermaid mindmap to SVG and optional PNG.")
    parser.add_argument("--input", required=True, help="Input .mmd file path")
    parser.add_argument("--output-svg", required=True, help="Output .svg file path")
    parser.add_argument("--output-png", default="", help="Optional output .png file path")
    parser.add_argument("--config", default="", help="Optional Mermaid JSON config path")
    parser.add_argument("--scale", type=int, default=2, help="PNG scale factor (default 2)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Input Mermaid file not found: {input_path}", file=sys.stderr)
        return 2

    mmdc = shutil.which("mmdc")
    if not mmdc:
        print("mmdc (Mermaid CLI) not found in PATH. Please prepare @mermaid-js/mermaid-cli in Runtime.", file=sys.stderr)
        return 3

    default_cfg = Path(__file__).resolve().parent.parent / "references" / "mermaid-config.json"
    mermaid_cfg = Path(args.config) if args.config else default_cfg
    puppeteer_cfg = _find_puppeteer_config()

    svg_path = Path(args.output_svg)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    _run_mmdc(mmdc, input_path, svg_path, mermaid_config=mermaid_cfg, puppeteer_config=puppeteer_cfg)

    if not svg_path.is_file() or svg_path.stat().st_size == 0:
        print("SVG output was not created or is empty.", file=sys.stderr)
        return 4

    svg_text = svg_path.read_text(encoding="utf-8", errors="strict")
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        print(f"Generated SVG is not valid XML: {exc}", file=sys.stderr)
        return 5
    if not root.tag.endswith("svg"):
        print(f"Unexpected SVG root tag: {root.tag}", file=sys.stderr)
        return 5

    result: dict[str, object] = {
        "ok": True,
        "input": str(input_path),
        "svg": str(svg_path),
        "svg_bytes": svg_path.stat().st_size,
    }

    if args.output_png:
        png_path = Path(args.output_png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        _run_mmdc(
            mmdc,
            input_path,
            png_path,
            mermaid_config=mermaid_cfg,
            puppeteer_config=puppeteer_cfg,
            scale=max(1, min(args.scale, 4)),
        )
        png_bytes = png_path.read_bytes()
        if len(png_bytes) < 24 or png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
            print("Generated PNG header is invalid.", file=sys.stderr)
            return 6
        width, height = struct.unpack(">II", png_bytes[16:24])
        result["png"] = str(png_path)
        result["png_bytes"] = len(png_bytes)
        result["png_width"] = width
        result["png_height"] = height

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
