#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Qwen3.5-9B Console Chatbot  —  VRAM-Aware Rewrite v2
=====================================================

VRAM management philosophy
---------------------------
  1. <think> blocks are DISPLAYED but NEVER stored in history.
     Storing think blocks caused OOM by turn 3-4 on a 16 GB GPU.

  2. Assistant responses are TRUNCATED before storage.
     Even without <think>, a 2500-token HTML file stored as history
     causes prompt tokens to explode (observed: 11355 tokens with only
     2 rounds of code generation history). HISTORY_STORE_CHARS caps
     how many characters of an assistant reply enter history.
     Default: 800 chars (~200 tokens). Enough to give the model
     context about what it last said, without blowing the budget.

  3. History is kept to MAX_ROUNDS recent user/assistant pairs (default 2).
     max_rounds * 2 turns + system prompt + new user turn must fit
     comfortably within the 8000-token soft budget.

  4. torch.cuda.empty_cache() + gc.collect() after EVERY generation.

  5. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set at startup.
     (Ignored on Windows with a UserWarning — harmless, no effect needed
     since the main fixes are items 1-4 above.)

  6. atexit + SIGINT handler for clean CUDA context release on exit.

  7. Token budget printed before every generation.

Run:
  .\.venv\Scripts\python.exe -u .\qwen35_console_chatbot_streaming.py \
      --model .\models\qwen3.5-9b --load_in_4bit --max_new_tokens 2500 --mode code

Commands:
  /help
  /exit  /quit
  /mode chat|code
  /think on|off
  /stream on|off
  /max_new N
  /max_rounds N          - recent turn pairs to keep in context (default 2)
  /store_chars N         - max chars stored per assistant turn (default 800)
  /tokens                - print current prompt token count
  /vram                  - print VRAM stats
  /reset                 - clear history
  /save file.json
  /load file.json
  /image <path> [prompt]
"""

import argparse
import atexit
import gc
import json
import os
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set BEFORE torch is imported.
# expandable_segments reduces reserved VRAM on Linux/Mac.
# On Windows it logs a UserWarning and is ignored — that is fine.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoVLModel  # type: ignore
except Exception:
    from transformers import AutoModelForVision2Seq as AutoVLModel  # type: ignore

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

try:
    from transformers import TextIteratorStreamer
    _HAS_STREAMER = True
except Exception:
    _HAS_STREAMER = False


# ── Helpers ────────────────────────────────────────────────────────────────────

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    s = THINK_RE.sub("", text) if text else text
    s = re.sub(r"^\s*(system|user|assistant)\s*\n", "", s, flags=re.IGNORECASE)
    return s.strip()


def truncate_for_storage(text: str, max_chars: int) -> str:
    """
    Truncate assistant text before storing in history.

    Why: Even without <think>, a 2500-token code block stored verbatim
    causes prompt tokens to grow to 11k+ with only 2 history rounds.
    We keep the first `max_chars` characters so the model knows what
    it last produced without blowing the token budget.
    """
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated for context window]"


def ensure_utf8_stdout():
    try:
        import io
        if sys.stdout and hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        if sys.stderr and hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
    except Exception:
        pass


def now_ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str):
    print(f"[{now_ts()}] {msg}", flush=True)


def log_vram(label: str = "") -> dict:
    if not torch.cuda.is_available():
        return {}
    alloc = torch.cuda.memory_allocated() / 1024**3
    resv  = torch.cuda.memory_reserved()   / 1024**3
    tag   = f" [{label}]" if label else ""
    log(f"VRAM{tag}: allocated={alloc:.2f} GB  reserved={resv:.2f} GB")
    return {"allocated_gb": alloc, "reserved_gb": resv, "label": label}


def cuda_release(model=None, processor=None):
    log("Releasing CUDA resources...")
    try:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            log_vram("released")
        log("CUDA context released cleanly.")
    except Exception as e:
        log(f"Warning during CUDA release: {e}")


def post_generate_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Data ───────────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str
    content: Any


# ── Bot ────────────────────────────────────────────────────────────────────────

class QwenBot:
    def __init__(
        self,
        model_path: str,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        attn_implementation: str = "eager",
        dtype: str = "float16",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1024 * 28 * 28,
        max_new_tokens: int = 2500,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        max_rounds: int = 2,
        history_store_chars: int = 800,   # ← NEW: caps stored assistant text
        enable_stream: bool = True,
        show_thinking: bool = True,
        mode: str = "chat",
    ):
        self.model_path          = model_path
        self.load_in_4bit        = load_in_4bit
        self.load_in_8bit        = load_in_8bit
        self.device_map          = device_map
        self.attn_implementation = attn_implementation
        self.dtype               = dtype
        self.min_pixels          = min_pixels
        self.max_pixels          = max_pixels
        self.max_new_tokens      = max_new_tokens
        self.do_sample           = do_sample
        self.temperature         = temperature
        self.top_p               = top_p
        self.top_k               = top_k
        self.repetition_penalty  = repetition_penalty
        self.max_rounds          = max_rounds
        self.history_store_chars = history_store_chars
        self.enable_stream       = enable_stream
        self.show_thinking       = show_thinking
        self.mode                = mode

        self.processor: Any = None
        self.model:     Any = None
        self._bad_words_ids_cache = None
        self.history: List[Turn] = []

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        log("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            trust_remote_code=True,
        )

        quant_cfg = None
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot use load_in_4bit and load_in_8bit together. Pick one.")
        elif self.load_in_4bit:
            if not _HAS_BNB:
                raise RuntimeError("bitsandbytes not available.")
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self.load_in_8bit:
            if not _HAS_BNB:
                raise RuntimeError("bitsandbytes not available.")
            quant_cfg = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        log("Loading model...")
        dtype_map = {
            "float16": torch.float16, "fp16": torch.float16,
            "bf16": torch.bfloat16,   "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype_obj = dtype_map.get(self.dtype.lower(), torch.float16)

        self.model = AutoVLModel.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            attn_implementation=self.attn_implementation,
            quantization_config=quant_cfg,
            dtype=dtype_obj,
            trust_remote_code=True,
        )
        log_vram("after load")
        # Release staging buffers that from_pretrained() leaves in the pool.
        # These are idle after loading completes but keep reserved VRAM at
        # ~15 GB. empty_cache() drops reserved back to ~8 GB before the
        # first generation, freeing ~6-7 GB of headroom for longer outputs.
        gc.collect()
        torch.cuda.empty_cache()
        log_vram("after load cleanup")
        self._bad_words_ids_cache = self._build_bad_words()

    def cleanup(self):
        model, proc = self.model, self.processor
        self.model = self.processor = None
        cuda_release(model=model, processor=proc)

    # ── Context ───────────────────────────────────────────────────────────────

    def _system_prompt(self) -> str:
        if self.mode == "code":
            return (
                "You are a coding assistant.\n"
                "Return ONLY the final code/content. No explanations, no prose.\n"
                "Do NOT use markdown fences (no ```).\n"
                "ALWAYS declare a filename header before every code block, even for a single file.\n"
                "Use exactly this format:\n"
                "----- filename.ext -----\n"
                "(file content here)\n"
                "For multiple files, repeat the header pattern for each file.\n"
                "The filename must be a real, usable filename with the correct extension."
            ).strip()
        return (
            "You are a helpful assistant.\n"
            "You may think in <think>...</think> if you want."
        ).strip()

    def _build_messages(self, user_turn: Turn) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()}
        ]
        for t in self.history:
            msgs.append({"role": t.role, "content": t.content})
        msgs.append({"role": user_turn.role, "content": user_turn.content})
        return msgs

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        assert self.processor is not None
        try:
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return len(self.processor.tokenizer.encode(prompt, add_special_tokens=False))
        except Exception:
            return 0

    def _trim_history(self):
        max_turns = self.max_rounds * 2
        if len(self.history) > max_turns:
            dropped = len(self.history) - max_turns
            self.history = self.history[-max_turns:]
            log(f"History trimmed: -{dropped} turns, keeping {len(self.history)}.")

    def _build_bad_words(self):
        assert self.processor is not None
        tok = self.processor.tokenizer
        ids = []
        for s in ["```", "```html", "```css", "```json", "```javascript", "```python"]:
            try:
                seq = tok.encode(s, add_special_tokens=False)
                if seq:
                    ids.append(seq)
            except Exception:
                pass
        return ids if ids else None

    # ── Inputs ────────────────────────────────────────────────────────────────

    def _get_device(self) -> torch.device:
        assert self.model is not None
        for p in self.model.parameters():
            if hasattr(p, "device") and p.device.type != "meta":
                return p.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_inputs(self, messages, image):
        assert self.processor is not None
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if image is not None:
            inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        else:
            inputs = self.processor(text=[prompt_text], return_tensors="pt")
        device = self._get_device()
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        return inputs

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate(self, messages, image, max_new_tokens) -> str:
        assert self.model is not None
        assert self.processor is not None

        inputs     = self._prepare_inputs(messages, image)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: Dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        if self.do_sample:
            gen_kwargs.update(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        if self._bad_words_ids_cache:
            gen_kwargs["bad_words_ids"] = self._bad_words_ids_cache

        try:
            if (not self.enable_stream) or (not _HAS_STREAMER):
                with torch.inference_mode():
                    out = self.model.generate(**gen_kwargs)
                return self.processor.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

            streamer = TextIteratorStreamer(self.processor.tokenizer, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer
            chunks: List[str] = []
            exc_holder: List[Optional[Exception]] = [None]

            def _bg():
                try:
                    with torch.inference_mode():
                        self.model.generate(**gen_kwargs)
                except Exception as e:
                    exc_holder[0] = e

            t = threading.Thread(target=_bg, daemon=True)
            t.start()

            print("\n--- streaming start ---\n", flush=True)
            in_think = False
            for chunk in streamer:
                chunks.append(chunk)
                if self.show_thinking:
                    print(chunk, end="", flush=True)
                else:
                    combined = "".join(chunks)
                    if "<think>" in combined and "</think>" not in combined:
                        in_think = True
                    elif "</think>" in combined:
                        in_think = False
                    if not in_think:
                        print(chunk, end="", flush=True)
            print("\n\n--- streaming end ---\n", flush=True)

            t.join()
            if exc_holder[0] is not None:
                raise exc_holder[0]

            return "".join(chunks).strip()

        finally:
            post_generate_cleanup()
            log_vram("after generate")

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat(self, user_turn: Turn, image=None) -> str:
        self._trim_history()
        messages    = self._build_messages(user_turn)
        token_count = self.count_tokens(messages)
        log(
            f"Prompt tokens: ~{token_count}  |  max_new: {self.max_new_tokens}  "
            f"|  history_turns: {len(self.history)}  "
            f"|  store_chars: {self.history_store_chars}"
        )

        t0  = time.time()
        raw = self._generate(messages, image=image, max_new_tokens=self.max_new_tokens)
        dt  = time.time() - t0

        if not self.enable_stream:
            print((raw if self.show_thinking else strip_think(raw)), flush=True)

        log(f"Done in {dt:.1f}s")

        # Strip <think> AND truncate long code outputs before storing
        stored = strip_think(raw)
        stored = truncate_for_storage(stored, self.history_store_chars)

        self.history.append(user_turn)
        self.history.append(Turn(role="assistant", content=stored))

        return raw

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        Path(path).write_text(
            json.dumps({"history": [asdict(t) for t in self.history]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log(f"Saved to {path}")

    def load_file(self, path: str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.history = [Turn(**t) for t in data.get("history", [])]
        self._trim_history()
        log(f"Loaded from {path}  (turns={len(self.history)})")

    def reset(self):
        self.history = []
        log("History cleared.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Qwen3.5 VRAM-aware console chatbot")
    ap.add_argument("--model",          required=True)
    ap.add_argument("--load_in_4bit",   action="store_true")
    ap.add_argument("--load_in_8bit",   action="store_true")
    ap.add_argument("--device_map",     default="auto")
    ap.add_argument("--dtype",          default="float16")
    ap.add_argument("--attn",           default="eager")
    ap.add_argument("--max_new_tokens", type=int,   default=2500)
    ap.add_argument("--max_rounds",     type=int,   default=2)
    ap.add_argument("--store_chars",    type=int,   default=800,
                    help="Max chars of assistant reply stored in history (default 800)")
    ap.add_argument("--temperature",    type=float, default=0.8)
    ap.add_argument("--top_p",          type=float, default=0.9)
    ap.add_argument("--top_k",          type=int,   default=50)
    ap.add_argument("--min_pixels",     type=int,   default=256 * 28 * 28)
    ap.add_argument("--max_pixels",     type=int,   default=1024 * 28 * 28)
    ap.add_argument("--no_stream",      action="store_true")
    ap.add_argument("--hide_think",     action="store_true")
    ap.add_argument("--deterministic",  action="store_true")
    ap.add_argument("--mode",           default="chat", choices=["chat", "code"])
    ap.add_argument("--memory_file",    default="")
    return ap.parse_args()


def main():
    ensure_utf8_stdout()
    args = parse_args()

    bot = QwenBot(
        model_path          = args.model,
        load_in_4bit        = args.load_in_4bit,
        load_in_8bit        = args.load_in_8bit,
        device_map          = args.device_map,
        attn_implementation = args.attn,
        dtype               = args.dtype,
        min_pixels          = args.min_pixels,
        max_pixels          = args.max_pixels,
        max_new_tokens      = args.max_new_tokens,
        do_sample           = not args.deterministic,
        temperature         = args.temperature,
        top_p               = args.top_p,
        top_k               = args.top_k,
        max_rounds          = args.max_rounds,
        history_store_chars = args.store_chars,
        enable_stream       = not args.no_stream,
        show_thinking       = not args.hide_think,
        mode                = args.mode,
    )

    atexit.register(bot.cleanup)

    def _sig(sig, frame):
        print("", flush=True)
        log("Interrupted — releasing CUDA before exit...")
        bot.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    bot.load()

    if args.memory_file:
        try:
            bot.load_file(args.memory_file)
        except Exception as e:
            log(f"Could not load memory file: {e}")

    print(
        f"\nQwen3.5 ready.  mode={bot.mode}  max_rounds={bot.max_rounds}  "
        f"store_chars={bot.history_store_chars}  max_new_tokens={bot.max_new_tokens}\n"
        "Type /help for commands.\n",
        flush=True,
    )

    while True:
        try:
            line = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("", flush=True)
            break

        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=2)
            cmd   = parts[0].lower()

            if cmd in ("/exit", "/quit"):
                break
            elif cmd == "/help":
                print(
                    "\nCommands:\n"
                    "  /exit  /quit\n"
                    "  /reset\n"
                    "  /mode chat|code\n"
                    "  /think on|off      (never stored in history)\n"
                    "  /stream on|off\n"
                    "  /max_new N\n"
                    "  /max_rounds N      recent turn pairs to keep\n"
                    "  /store_chars N     max chars stored per assistant turn\n"
                    "  /tokens            current prompt token count\n"
                    "  /vram\n"
                    "  /save file.json\n"
                    "  /load file.json\n"
                    "  /image <path> [prompt]\n",
                    flush=True,
                )
            elif cmd == "/reset":
                bot.reset()
            elif cmd == "/mode" and len(parts) >= 2:
                if parts[1] in ("chat", "code"):
                    bot.mode = parts[1]; log(f"mode={bot.mode}")
                else:
                    log("Usage: /mode chat|code")
            elif cmd == "/think" and len(parts) >= 2:
                bot.show_thinking = parts[1].lower() == "on"
                log(f"show_thinking={bot.show_thinking}")
            elif cmd == "/stream" and len(parts) >= 2:
                bot.enable_stream = parts[1].lower() == "on"
                log(f"stream={bot.enable_stream}")
            elif cmd == "/max_new" and len(parts) >= 2:
                try:
                    bot.max_new_tokens = int(parts[1]); log(f"max_new_tokens={bot.max_new_tokens}")
                except ValueError:
                    log("Usage: /max_new 2500")
            elif cmd == "/max_rounds" and len(parts) >= 2:
                try:
                    bot.max_rounds = int(parts[1]); bot._trim_history()
                    log(f"max_rounds={bot.max_rounds}  history={len(bot.history)}")
                except ValueError:
                    log("Usage: /max_rounds 2")
            elif cmd == "/store_chars" and len(parts) >= 2:
                try:
                    bot.history_store_chars = int(parts[1])
                    log(f"store_chars={bot.history_store_chars}")
                except ValueError:
                    log("Usage: /store_chars 800")
            elif cmd == "/tokens":
                dummy = Turn(role="user", content="(estimate)")
                tok   = bot.count_tokens(bot._build_messages(dummy))
                log(f"Current prompt ~{tok} tokens  (history={len(bot.history)} turns)")
            elif cmd == "/vram":
                log_vram("manual")
            elif cmd == "/save" and len(parts) >= 2:
                bot.save(parts[1])
            elif cmd == "/load" and len(parts) >= 2:
                bot.load_file(parts[1])
            elif cmd == "/image" and len(parts) >= 2:
                img_path = parts[1]
                prompt_text = parts[2].strip() if len(parts) == 3 else "Describe this image."
                p = Path(img_path)
                if not p.exists():
                    log(f"Image not found: {p}"); continue
                try:
                    img = Image.open(str(p)).convert("RGB")
                except Exception as e:
                    log(f"Failed to open image: {e}"); continue
                user_turn = Turn(role="user", content=[
                    {"type": "image", "image": img},
                    {"type": "text",  "text": prompt_text},
                ])
                log(f"Image: {p.name}  {img.size[0]}x{img.size[1]}")
                bot.chat(user_turn, image=img)
            else:
                log(f"Unknown command: {cmd}  — type /help")
            continue

        bot.chat(Turn(role="user", content=line))

    log("Bye.")


if __name__ == "__main__":
    main()