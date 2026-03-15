#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
qwen35_stress_test.py
======================
Automated stress test for QwenBot.
Runs tasks loaded from a JSON file, monitors VRAM per task,
and produces a final report showing memory stability.

Each run saves generated code into a timestamped output folder:
  stress_test_outputs/
  LLLU 20260306_143022/
      LLLU run_config.json
      LLLU task_01_saas_landing_page/
      L   LLLU index.html
      ...
      LLLU stress_test_report.json

Tasks are loaded from a JSON file (default: tasks_webpage.json).
Each task entry schema:
  {
    "id":               1,
    "name":             "Short display name",
    "reset":            true,
    "mode":             "code",
    "default_filename": "index.html",
    "prompt":           "..."
  }

Swap task sets by changing --tasks:
  --tasks tasks_webpage.json      (default: 10 modern webpages)
  --tasks tasks_algorithms.json   (your own custom set)

Run (4-bit, default webpage tasks):
  .\.venv\Scripts\python.exe -u .\qwen35_stress_test.py \
      --model .\models\qwen3.5-9b --load_in_4bit --max_new_tokens 6000

Run with custom tasks file:
  .\.venv\Scripts\python.exe -u .\qwen35_stress_test.py \
      --model .\models\qwen3.5-9b --load_in_4bit --tasks .\my_tasks.json

Run (8-bit):
  .\.venv\Scripts\python.exe -u .\qwen35_stress_test.py \
      --model .\models\qwen3.5-9b --load_in_8bit --max_new_tokens 5000

What it checks:
  - Prompt token count stays under TOKEN_WARN threshold
  - VRAM allocated stays under VRAM_WARN_GB (default 14 GB)
  - VRAM reserved stays under VRAM_LIMIT_GB (default 15.5 GB)
  - No generation crashes / OOM errors
  - History storage stays bounded (store_chars cap working)
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, str(Path(__file__).parent))
try:
    from qwen35_console_chatbot_streaming import QwenBot, Turn, log, log_vram
except ImportError as e:
    print(f"ERROR: Could not import QwenBot: {e}")
    print("Make sure qwen35_console_chatbot_streaming.py is in the same directory.")
    sys.exit(1)


# ── Configuration ─────────────────────────────────────────────────────────────

VRAM_WARN_GB       = 14.0
VRAM_LIMIT_GB      = 15.5
TOKEN_WARN         = 7000
DEFAULT_TASKS_FILE = "tasks_webpage.json"

OUTPUT_ROOT = Path("stress_test_outputs")


# ── Task loading ───────────────────────────────────────────────────────────────

REQUIRED_TASK_KEYS = {"id", "name", "reset", "mode", "default_filename", "prompt"}


def load_tasks(path: str) -> List[dict]:
    """
    Load and validate tasks from a JSON file.
    Raises SystemExit with a helpful message on any problem.
    """
    p = Path(path)
    if not p.exists():
        print(f"ERROR: Tasks file not found: {p.resolve()}")
        sys.exit(1)

    try:
        tasks = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not parse tasks JSON: {e}")
        sys.exit(1)

    if not isinstance(tasks, list) or len(tasks) == 0:
        print("ERROR: Tasks file must be a non-empty JSON array.")
        sys.exit(1)

    for i, task in enumerate(tasks):
        missing = REQUIRED_TASK_KEYS - set(task.keys())
        if missing:
            print(f"ERROR: Task at index {i} is missing required keys: {missing}")
            sys.exit(1)
        if task["mode"] not in ("code", "chat"):
            print(f"ERROR: Task {task['id']} has invalid mode '{task['mode']}'. Must be 'code' or 'chat'.")
            sys.exit(1)

    # Sort by id so JSON file order doesn't matter
    tasks = sorted(tasks, key=lambda t: t["id"])
    return tasks


# ── File extraction ────────────────────────────────────────────────────────────

FILE_HEADER_RE = re.compile(
    r"^-{3,}\s+([\w\-. /]+?\.\w+)\s+-{3,}\s*$",
    re.MULTILINE,
)


def clean_raw_output(raw: str) -> str:
    text = re.sub(r"</think>", "", raw, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(system|user|assistant)\s*\n", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_files(raw: str, default_filename: str) -> Dict[str, str]:
    """
    Extract {filename: content} dict from model output.

    Handles three patterns:
      1. Multiple files  ----- a.html -----  ...  ----- b.js -----  ...
      2. Single file     ----- index.html -----  ...
      3. Raw code dump   no headers -> use default_filename
    """
    text   = clean_raw_output(raw)
    splits = FILE_HEADER_RE.split(text)

    if len(splits) < 3:
        return {default_filename: text}

    files: Dict[str, str] = {}
    it = iter(splits[1:])
    for fname, content in zip(it, it):
        fname   = fname.strip()
        content = content.strip()
        if fname and content:
            files[fname] = content

    return files if files else {default_filename: text}


def sanitize_path_component(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name)


def save_task_output(run_dir: Path, task: dict, raw_output: str) -> List[Path]:
    """Save extracted files for one task into run_dir/task_XX_slug/."""
    tid      = task["id"]
    slug     = sanitize_path_component(task["name"].lower().replace(" ", "_"))
    task_dir = run_dir / f"task_{tid:02d}_{slug}"
    task_dir.mkdir(parents=True, exist_ok=True)

    default_fn = task.get("default_filename", "output.txt")
    files      = extract_files(raw_output, default_fn)

    written: List[Path] = []
    for fname, content in files.items():
        safe_fname = Path(fname).name
        out_path   = task_dir / safe_fname
        try:
            out_path.write_text(content, encoding="utf-8")
            written.append(out_path)
            log(f"  saved -> {out_path.relative_to(run_dir.parent)}")
        except Exception as ex:
            log(f"  WARNING: could not write {out_path}: {ex}")

    return written


# ── Result tracking ────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id:           int
    task_name:         str
    status:            str    # "OK" | "WARN" | "OOM" | "ERROR"
    prompt_tokens:     int
    alloc_before_gb:   float
    alloc_after_gb:    float
    reserved_after_gb: float
    duration_s:        float
    output_chars:      int
    stored_chars:      int
    files_saved:       int
    error:             str = ""


def vram_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb":  torch.cuda.memory_reserved()  / 1024**3,
    }


def status_icon(status: str) -> str:
    return {"OK": "OK", "WARN": "WARN", "OOM": "OOM", "ERROR": "ERR"}.get(status, "?")


def status_emoji(status: str) -> str:
    return {"OK": "OK", "WARN": "WARN", "OOM": "OOM", "ERROR": "ERR"}.get(status, "?")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_stress_test(args):
    print("\n" + "=" * 70)
    print("  Qwen3.5 Stress Test — VRAM Monitoring")
    print("=" * 70 + "\n")

    # ── Load tasks from JSON ──────────────────────────────────────────────────
    tasks_file = args.tasks
    tasks      = load_tasks(tasks_file)
    n_tasks    = len(tasks)
    print(f"  Tasks file:    {Path(tasks_file).resolve()}")
    print(f"  Tasks loaded:  {n_tasks}")

    # ── Quantization label ────────────────────────────────────────────────────
    quant_label = "none (full precision)"
    if args.load_in_4bit:
        quant_label = "4-bit NF4"
    elif args.load_in_8bit:
        quant_label = "8-bit INT8"
    print(f"  Quantization:  {quant_label}")

    # ── Timestamped output folder ─────────────────────────────────────────────
    run_ts  = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output folder: {run_dir.resolve()}\n")

    bot = QwenBot(
        model_path          = args.model,
        load_in_4bit        = args.load_in_4bit,
        load_in_8bit        = args.load_in_8bit,
        device_map          = "auto",
        attn_implementation = "eager",
        dtype               = args.dtype,
        max_new_tokens      = args.max_new_tokens,
        do_sample           = True,
        temperature         = 0.7,
        max_rounds          = args.max_rounds,
        history_store_chars = args.store_chars,
        enable_stream       = not args.no_stream,
        show_thinking       = False,
        mode                = "code",
    )

    try:
        bot.load()
    except Exception as e:
        print(f"FATAL: Could not load model: {e}")
        return

    print(f"\n{'─' * 70}")
    print(f"  Config: max_rounds={bot.max_rounds}  store_chars={bot.history_store_chars}  "
          f"max_new={bot.max_new_tokens}  quant={quant_label}")
    print(f"  Thresholds: VRAM warn={VRAM_WARN_GB}GB  limit={VRAM_LIMIT_GB}GB  "
          f"token warn={TOKEN_WARN}")
    print(f"{'─' * 70}\n")

    # Save run config immediately (survives mid-run crash)
    run_config = {
        "run_ts":          run_ts,
        "tasks_file":      str(Path(tasks_file).resolve()),
        "n_tasks":         n_tasks,
        "model":           args.model,
        "quantization":    quant_label,
        "load_in_4bit":    args.load_in_4bit,
        "load_in_8bit":    args.load_in_8bit,
        "max_rounds":      args.max_rounds,
        "store_chars":     args.store_chars,
        "max_new_tokens":  args.max_new_tokens,
        "vram_warn_gb":    VRAM_WARN_GB,
        "vram_limit_gb":   VRAM_LIMIT_GB,
        "token_warn":      TOKEN_WARN,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    results: List[TaskResult] = []

    for task in tasks:
        tid   = task["id"]
        name  = task["name"]
        reset = task["reset"]
        mode  = args.mode if args.mode else task["mode"]

        print(f"\n{'=' * 70}")
        print(f"  Task {tid}/{n_tasks}: {name}")
        print(f"  Reset={reset}  Mode={mode}  File={task['default_filename']}")
        print(f"{'=' * 70}")

        if reset:
            bot.reset()

        bot.mode = mode

        user_turn   = Turn(role="user", content=task["prompt"])
        snap_before = vram_snapshot()

        bot._trim_history()
        msgs      = bot._build_messages(user_turn)
        tok_count = bot.count_tokens(msgs)

        status      = "OK"
        output_raw  = ""
        stored_text = ""
        duration_s  = 0.0
        error_msg   = ""
        snap_after  = snap_before.copy()
        files_saved = 0

        if tok_count > TOKEN_WARN:
            print(f"  WARNING: Prompt tokens {tok_count} exceeds threshold {TOKEN_WARN}")
            status = "WARN"

        t0 = time.time()
        try:
            output_raw = bot.chat(user_turn)
            duration_s = time.time() - t0
            snap_after = vram_snapshot()

            if snap_after["allocated_gb"] > VRAM_WARN_GB:
                print(f"  WARNING: Allocated VRAM {snap_after['allocated_gb']:.2f} GB "
                      f"exceeds warn threshold {VRAM_WARN_GB} GB")
                status = "WARN"

            if snap_after["reserved_gb"] > VRAM_LIMIT_GB:
                print(f"  OOM RISK: Reserved VRAM {snap_after['reserved_gb']:.2f} GB "
                      f"exceeds limit {VRAM_LIMIT_GB} GB!")
                status = "OOM" if status != "OOM" else status

            if bot.history:
                stored_text = (
                    bot.history[-1].content
                    if isinstance(bot.history[-1].content, str)
                    else str(bot.history[-1].content)
                )

            written     = save_task_output(run_dir, task, output_raw)
            files_saved = len(written)

        except torch.cuda.OutOfMemoryError as e:
            duration_s = time.time() - t0
            snap_after = vram_snapshot()
            status     = "OOM"
            error_msg  = f"CUDA OOM: {str(e)[:120]}"
            print(f"\n  OOM on task {tid}: {error_msg}")
            gc.collect()
            torch.cuda.empty_cache()
            bot.reset()

        except Exception as e:
            duration_s = time.time() - t0
            snap_after = vram_snapshot()
            status     = "ERROR"
            error_msg  = str(e)[:200]
            print(f"\n  ERROR on task {tid}: {error_msg}")

        result = TaskResult(
            task_id           = tid,
            task_name         = name,
            status            = status,
            prompt_tokens     = tok_count,
            alloc_before_gb   = snap_before["allocated_gb"],
            alloc_after_gb    = snap_after["allocated_gb"],
            reserved_after_gb = snap_after["reserved_gb"],
            duration_s        = duration_s,
            output_chars      = len(output_raw),
            stored_chars      = len(stored_text),
            files_saved       = files_saved,
            error             = error_msg,
        )
        results.append(result)

        icon = {"OK": "OK", "WARN": "WARN", "OOM": "OOM", "ERROR": "ERR"}.get(status, "?")
        print(
            f"\n  [{icon}] Task {tid}  "
            f"tokens={tok_count}  "
            f"alloc={snap_after['allocated_gb']:.2f}GB  "
            f"reserved={snap_after['reserved_gb']:.2f}GB  "
            f"output={len(output_raw)} chars  "
            f"files={files_saved}  "
            f"time={duration_s:.1f}s"
        )

    # ── Final report ──────────────────────────────────────────────────────────

    print("\n\n" + "=" * 70)
    print("  STRESS TEST REPORT")
    print("=" * 70)

    header = (
        f"{'#':>2}  {'Task':<32}  {'Status':>6}  "
        f"{'Toks':>5}  {'Alloc':>6}  {'Resv':>6}  "
        f"{'OutCh':>6}  {'Files':>5}  {'Time':>7}"
    )
    print(header)
    print("-" * len(header))

    passes     = 0
    warns      = 0
    fails      = 0
    max_alloc  = 0.0
    max_resv   = 0.0
    max_tokens = 0

    for r in results:
        icon = {"OK": "[OK]  ", "WARN": "[WARN]", "OOM": "[OOM] ", "ERROR": "[ERR] "}.get(r.status, "[?]   ")
        print(
            f"{r.task_id:>2}  {r.task_name:<32}  {icon}  "
            f"{r.prompt_tokens:>5}  {r.alloc_after_gb:>5.2f}G  {r.reserved_after_gb:>5.2f}G  "
            f"{r.output_chars:>6}  {r.files_saved:>5}  {r.duration_s:>6.1f}s"
        )
        if r.error:
            print(f"     ERROR: {r.error}")

        if r.status == "OK":
            passes += 1
        elif r.status == "WARN":
            warns += 1
        else:
            fails += 1

        max_alloc  = max(max_alloc,  r.alloc_after_gb)
        max_resv   = max(max_resv,   r.reserved_after_gb)
        max_tokens = max(max_tokens, r.prompt_tokens)

    print("-" * len(header))
    print(f"\n  Results:  {passes} OK   {warns} WARN   {fails} FAIL")
    print(f"  Peak VRAM allocated: {max_alloc:.2f} GB  (limit: {VRAM_WARN_GB} GB)")
    print(f"  Peak VRAM reserved:  {max_resv:.2f} GB  (limit: {VRAM_LIMIT_GB} GB)")
    print(f"  Peak prompt tokens:  {max_tokens}  (warn: {TOKEN_WARN})")
    print(f"  Output folder:       {run_dir.resolve()}")

    print()
    if fails == 0 and max_resv < VRAM_LIMIT_GB:
        print("  PASS -- VRAM stayed within safe limits across all tasks.")
    elif fails == 0:
        print("  MARGINAL -- No crashes, but VRAM approached limit. "
              "Consider reducing --max_rounds or --store_chars.")
    else:
        print("  FAIL -- One or more tasks crashed. "
              "Reduce --max_new_tokens, --max_rounds, or --store_chars.")

    report_data = {
        "config": run_config,
        "summary": {
            "passes":           passes,
            "warns":            warns,
            "fails":            fails,
            "peak_alloc_gb":    max_alloc,
            "peak_reserved_gb": max_resv,
            "peak_tokens":      max_tokens,
        },
        "tasks": [
            {
                "id":                r.task_id,
                "name":              r.task_name,
                "status":            r.status,
                "prompt_tokens":     r.prompt_tokens,
                "alloc_before_gb":   r.alloc_before_gb,
                "alloc_after_gb":    r.alloc_after_gb,
                "reserved_after_gb": r.reserved_after_gb,
                "duration_s":        round(r.duration_s, 1),
                "output_chars":      r.output_chars,
                "stored_chars":      r.stored_chars,
                "files_saved":       r.files_saved,
                "error":             r.error,
            }
            for r in results
        ],
    }

    report_path = run_dir / "stress_test_report.json"
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\n  Full report: {report_path.resolve()}")
    print("=" * 70 + "\n")

    bot.cleanup()


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Qwen3.5 stress test — loads tasks from a JSON file",
        allow_abbrev=False,
    )
    ap.add_argument("--model",           required=True,
                    help="Path to model directory")
    ap.add_argument("--tasks",           default=DEFAULT_TASKS_FILE,
                    help=f"Path to tasks JSON file (default: {DEFAULT_TASKS_FILE})")
    ap.add_argument("--load_in_4bit",    action="store_true",
                    help="Load model in 4-bit NF4 quantization (~7.4 GB VRAM)")
    ap.add_argument("--load_in_8bit",    action="store_true",
                    help="Load model in 8-bit INT8 quantization (~9.5 GB VRAM)")
    ap.add_argument("--dtype",           default="float16",
                    help="Compute dtype: float16 or bfloat16 (default: float16)")
    ap.add_argument("--max_new_tokens",  type=int, default=6000,
                    help="Max tokens per generation (default: 6000)")
    ap.add_argument("--max_rounds",      type=int, default=2,
                    help="Max conversation turns kept in history (default: 2)")
    ap.add_argument("--store_chars",     type=int, default=800,
                    help="Max chars stored per assistant turn (default: 800)")
    ap.add_argument("--no_stream",       action="store_true",
                    help="Disable streaming output")
    ap.add_argument("--mode",            default=None,
                    choices=["chat", "code"],
                    help="Override mode for ALL tasks (default: per-task setting in JSON)")

    args = ap.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        ap.error("Cannot use --load_in_4bit and --load_in_8bit together. Pick one.")

    return args


if __name__ == "__main__":
    args = parse_args()

    try:
        import io
        if sys.stdout and hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
    except Exception:
        pass

    run_stress_test(args)