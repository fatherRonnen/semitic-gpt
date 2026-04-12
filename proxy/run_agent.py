#!/usr/bin/env python3
"""
Autonomous experiment agent for Hebrew GPT optimization (v2).

Improvements over v1:
  - Experiment snapshots saved to snapshots/exp_NNN_train.py
  - Results tracked in results.tsv
  - Agent prompt includes experiment memory (last 10 results + diffs)
  - Best snapshot tracking with rollback

Uses Claude Sonnet 4.6 via AWS Bedrock.
"""

import os
import sys
import csv
import json
import time
import shutil
import difflib
import subprocess
import re
from pathlib import Path
from datetime import datetime

import boto3

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
TRAIN_PY = BASE_DIR / "train.py"
SNAPSHOTS_DIR = BASE_DIR / "snapshots"
RESULTS_TSV = BASE_DIR / "results.tsv"
PROGRAM_MD = BASE_DIR / "program.md"

MAX_EXPERIMENTS = 50
TRAIN_TIMEOUT = 480  # 3 min (2 min training + buffer)
MODEL_ID = "us.anthropic.claude-sonnet-4-6"
BEDROCK_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

SNAPSHOTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Bedrock client
# ---------------------------------------------------------------------------

def get_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=BEDROCK_REGION,
        config=boto3.session.Config(read_timeout=300, connect_timeout=30),
    )


def call_claude(client, system_prompt: str, user_prompt: str) -> str:
    """Call Claude Sonnet 4.6 via Bedrock invoke_model API with extended thinking."""
    import json as _json
    body = _json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 32000,
        "temperature": 1,  # required for extended thinking
        "thinking": {
            "type": "enabled",
            "budget_tokens": 8000,
        },
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    })
    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = _json.loads(response["body"].read())
    # With thinking, content has thinking blocks then text blocks
    for block in result["content"]:
        if block.get("type") == "text":
            return block["text"]
    return result["content"][-1].get("text", "")


# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------

def load_results() -> list[dict]:
    """Load experiment results from TSV."""
    if not RESULTS_TSV.exists():
        return []
    results = []
    with open(RESULTS_TSV, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results.append(row)
    return results


def save_result(exp_num: int, val_bpb: float, status: str,
                description: str, snapshot_file: str):
    """Append a result to the TSV."""
    header_needed = not RESULTS_TSV.exists()
    with open(RESULTS_TSV, "a") as f:
        if header_needed:
            f.write("experiment\tval_bpb\tstatus\tdescription\tsnapshot_file\ttimestamp\n")
        f.write(f"{exp_num}\t{val_bpb:.4f}\t{status}\t{description}\t{snapshot_file}\t"
                f"{datetime.utcnow().isoformat()}\n")


def get_best_result(results: list[dict]) -> dict | None:
    """Return the result with lowest val_bpb."""
    kept = [r for r in results if r.get("status") == "kept"]
    if not kept:
        return None
    return min(kept, key=lambda r: float(r["val_bpb"]))


# ---------------------------------------------------------------------------
# Snapshots & diffs
# ---------------------------------------------------------------------------

def save_snapshot(exp_num: int) -> str:
    """Save current train.py as a snapshot. Returns snapshot filename."""
    filename = f"exp_{exp_num:03d}_train.py"
    dest = SNAPSHOTS_DIR / filename
    shutil.copy2(TRAIN_PY, dest)
    return filename


def load_snapshot(filename: str) -> str:
    """Load a snapshot file's content."""
    path = SNAPSHOTS_DIR / filename
    if path.exists():
        return path.read_text()
    return ""


def get_diff(text_a: str, text_b: str, label_a: str = "before", label_b: str = "after") -> str:
    """Unified diff between two texts."""
    lines_a = text_a.splitlines(keepends=True)
    lines_b = text_b.splitlines(keepends=True)
    diff = difflib.unified_diff(lines_a, lines_b, fromfile=label_a, tofile=label_b)
    return "".join(diff)


# ---------------------------------------------------------------------------
# Training execution
# ---------------------------------------------------------------------------

def run_training() -> tuple[float | None, str]:
    """Execute train.py and parse the val_bpb result.

    Returns (val_bpb, stdout). val_bpb is None if training failed.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        result = subprocess.run(
            ["python3", "-u", str(TRAIN_PY)],
            capture_output=True, text=True, timeout=TRAIN_TIMEOUT,
            cwd=str(BASE_DIR), env=env,
        )
        output = result.stdout + "\n" + result.stderr
        print(output[-3000:])  # Show last 3000 chars

        # Parse val_bpb from output (last occurrence)
        matches = re.findall(r"^val_bpb=([0-9.]+)", output, re.MULTILINE)
        if matches:
            return float(matches[-1]), output
        else:
            print("WARNING: Could not parse val_bpb from output")
            return None, output

    except subprocess.TimeoutExpired:
        print("ERROR: Training timed out")
        return None, "TIMEOUT"
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        return None, str(e)


# ---------------------------------------------------------------------------
# Agent prompt construction
# ---------------------------------------------------------------------------

def build_agent_prompt(results: list[dict], current_train: str,
                       best_snapshot: str | None) -> tuple[str, str]:
    """Build system and user prompts for the agent."""

    # Load program.md for system instructions
    system = ""
    if PROGRAM_MD.exists():
        system = PROGRAM_MD.read_text()

    # Build user prompt with full context
    parts = []

    # Current train.py
    parts.append("## Current train.py (this is what will be modified)\n")
    parts.append(f"```python\n{current_train}\n```\n")

    # Results history (last 10)
    if results:
        parts.append("## Experiment History (last 10)\n")
        parts.append("| # | val_bpb | status | description |")
        parts.append("|---|---------|--------|-------------|")
        for r in results[-10:]:
            parts.append(f"| {r['experiment']} | {r['val_bpb']} | {r['status']} | "
                         f"{r.get('description', 'N/A')} |")
        parts.append("")

        best = get_best_result(results)
        if best:
            parts.append(f"**Current best: exp_{best['experiment']} with val_bpb={best['val_bpb']}**\n")

    # Diffs from last 2-3 failed attempts vs best
    if best_snapshot and results:
        best_content = load_snapshot(best_snapshot)
        failed = [r for r in results if r.get("status") == "discarded"][-3:]
        if failed and best_content:
            parts.append("## Recent Failed Attempts (diffs from best)\n")
            for r in failed:
                snap = r.get("snapshot_file", "")
                if snap:
                    failed_content = load_snapshot(snap)
                    if failed_content:
                        diff = get_diff(best_content, failed_content,
                                        f"best_exp_{r.get('experiment', '?')}",
                                        f"failed_exp_{r['experiment']}")
                        if diff:
                            parts.append(f"### Exp {r['experiment']} "
                                         f"(bpb={r['val_bpb']}, {r.get('description', '')})")
                            parts.append(f"```diff\n{diff[:3000]}\n```\n")

    # Final instruction
    next_exp = len(results) + 1
    parts.append(f"\n## Your Task: Experiment {next_exp}")
    parts.append("""
Propose a modification to train.py to improve val_bpb (lower is better).
The morphology-aware tokenizer in prepare.py already handles Hebrew prefix splitting.

Focus on: TRAINING OPTIMIZATIONS and MULTI-SCALE ENSEMBLES. Do NOT change the base architecture (WIDTH/DEPTH/HEADS).

IMPORTANT RULES:
1. Output ONLY the complete modified train.py inside ```python ... ``` blocks
2. Include a one-line comment at the top: # EXPERIMENT: <description>
3. Keep the same output format (val_bpb=, training_seconds=, etc.)
4. Keep imports from prepare.py (DataLoader, evaluate_bpb, VOCAB_SIZE, etc.)
5. Stay within 24GB VRAM (A10G GPU), 32GB system RAM
6. Must complete in 5 minutes
7. DO NOT change data paths — data is at data/train_morphology.bin and data/val_morphology.bin
8. DO NOT use uv or any external package manager — use python3 directly
9. The working baseline uses: from prepare import (DataLoader, evaluate_bpb, VOCAB_SIZE, MAX_SEQ_LEN, DEVICE_BATCH_SIZE, TOKENIZER_MODE, DATA_DIR)
10. Always wrap torch.compile in try/except — it may fail on this system
11. DO NOT change architecture (WIDTH=896, DEPTH=6, N_HEADS=14) — focus ONLY on training method
12. TRY THESE APPROACHES (in order of expected impact):
    a. Muon optimizer (momentum + Newton's method) — can beat AdamW significantly
    b. Stochastic Weight Averaging (SWA) — average weights over last N% of training
    c. Cosine schedule with warm restarts — multiple annealing cycles
    d. Snapshot ensembles — save checkpoints at cycle minima, average logits at eval
    e. Multi-model ensemble — train 2-3 smaller models, average logits
    f. µP (maximal update parameterization) — per-layer LR scaling
    g. Gradient noise injection — add noise to gradients for regularization
    h. Label smoothing — soften targets to prevent overconfidence
    i. Mixed precision (bf16/fp16) — may allow faster iteration
    j. Exponential Moving Average (EMA) of model weights
""")

    return system, "\n".join(parts)


def extract_code(response: str) -> tuple[str | None, str]:
    """Extract Python code and description from Claude's response."""
    # Find code block
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if not match:
        return None, "no code found"

    code = match.group(1).strip()

    # Extract experiment description from comment
    desc_match = re.search(r"#\s*EXPERIMENT:\s*(.+)", code)
    description = desc_match.group(1).strip() if desc_match else "no description"

    return code, description


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Hebrew GPT AutoResearch Agent v2")
    print("=" * 60)

    client = get_bedrock_client()
    results = load_results()
    start_exp = len(results) + 1

    # Determine current best snapshot
    best = get_best_result(results)
    best_snapshot = best.get("snapshot_file") if best else None
    best_bpb = float(best["val_bpb"]) if best else float("inf")

    print(f"Starting from experiment {start_exp}")
    if best:
        print(f"Current best: exp_{best['experiment']} with val_bpb={best['val_bpb']}")

    for exp_num in range(start_exp, MAX_EXPERIMENTS + 1):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {exp_num}/{MAX_EXPERIMENTS}")
        print(f"{'='*60}")

        # Read current train.py
        current_train = TRAIN_PY.read_text()

        # Build prompt and call Claude
        system_prompt, user_prompt = build_agent_prompt(results, current_train, best_snapshot)

        print("Calling Claude for next experiment proposal...")
        try:
            response = call_claude(client, system_prompt, user_prompt)
        except Exception as e:
            print(f"ERROR calling Claude: {e}")
            time.sleep(30)
            continue

        # Extract code
        new_code, description = extract_code(response)
        if new_code is None:
            print(f"ERROR: Could not extract code from response")
            save_result(exp_num, 99.0, "error", "no code extracted", "")
            results = load_results()
            continue

        print(f"Experiment description: {description}")

        # Save snapshot of current train.py BEFORE overwriting
        if exp_num == 1 and not best_snapshot:
            # Save baseline
            baseline_snap = save_snapshot(0)

        # Write new code
        TRAIN_PY.write_text(new_code)

        # Save snapshot of THIS experiment
        snapshot_file = save_snapshot(exp_num)
        print(f"Saved snapshot: {snapshot_file}")

        # Run training
        print("Running training...")
        val_bpb, output = run_training()

        if val_bpb is None:
            print(f"Training failed, rolling back to best")
            # Rollback
            if best_snapshot:
                shutil.copy2(SNAPSHOTS_DIR / best_snapshot, TRAIN_PY)
            save_result(exp_num, 99.0, "error", f"FAILED: {description}", snapshot_file)
            results = load_results()
            continue

        print(f"\nResult: val_bpb = {val_bpb:.4f}")

        # Decide: keep or discard
        if val_bpb < best_bpb:
            print(f"✓ IMPROVEMENT! {best_bpb:.4f} → {val_bpb:.4f}")
            save_result(exp_num, val_bpb, "kept", description, snapshot_file)
            best_bpb = val_bpb
            best_snapshot = snapshot_file
        else:
            print(f"✗ No improvement ({val_bpb:.4f} >= {best_bpb:.4f}), rolling back")
            save_result(exp_num, val_bpb, "discarded", description, snapshot_file)
            # Rollback to best
            if best_snapshot:
                shutil.copy2(SNAPSHOTS_DIR / best_snapshot, TRAIN_PY)

        results = load_results()

    print(f"\n{'='*60}")
    print(f"DONE — {MAX_EXPERIMENTS} experiments completed")
    print(f"Best val_bpb: {best_bpb:.4f}")
    if best_snapshot:
        print(f"Best snapshot: {best_snapshot}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
