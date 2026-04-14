import csv
import random
from functools import partial
from pathlib import Path
from typing import List, Tuple

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

from config import DataConfig

DIFF_START = "<|diff|>"
DIFF_END = "<|endofdiff|>"
MSG_START = "<|msg|>"
MSG_END = "<|endofmsg|>"

SPECIAL_TOKENS = [DIFF_START, DIFF_END, MSG_START, MSG_END]

PAD_ID = 50256

_SPECIAL_TOKENS_SET = set(SPECIAL_TOKENS)


def make_tokenizer():
    base = tiktoken.get_encoding("gpt2")
    return tiktoken.Encoding(
        name="gpt2_diffai",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={
            **base._special_tokens,
            DIFF_START: 50257,
            DIFF_END: 50258,
            MSG_START: 50259,
            MSG_END: 50260,
        }
    )
IGNORE_INDEX = -100


def prompt_csv_path() -> Path:
    while True:
        raw = input("Enter path to commitbench_long.csv: ").strip().strip('"').strip("'")
        p = Path(raw)
        if p.is_file():
            return p
        print(f"File not found: {p}")


def load_commitbench(cfg: DataConfig) -> Tuple[List[dict], List[dict], List[dict]]:
    if cfg.csv_path is None or not Path(cfg.csv_path).is_file():
        cfg.csv_path = prompt_csv_path()

    train, val, test = [], [], []
    limits = {"train": cfg.max_rows_train, "val": cfg.max_rows_val, "test": cfg.max_rows_test}
    counts = {"train": 0, "val": 0, "test": 0}

    with open(cfg.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split not in limits:
                continue
            if counts[split] >= limits[split]:
                if all(counts[s] >= limits[s] for s in limits):
                    break
                continue
            diff_text = row["diff"].strip()
            msg_text = row["message"].strip()
            if not diff_text or not msg_text:
                continue
            entry = {"diff": diff_text, "message": msg_text}
            if split == "train":
                train.append(entry)
            elif split == "val":
                val.append(entry)
            else:
                test.append(entry)
            counts[split] += 1

    random.shuffle(train)
    print(f"Loaded: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def truncate_diff(diff_text: str, enc, max_tokens: int) -> str:
    """
    Fit a unified diff into *max_tokens* while preserving breadth over depth.

    Naive head-truncation would drop every file after the first one or two,
    giving the model no signal about the overall scope of a change.  This
    function instead:

    1. Splits the diff into per-file sections (on ``diff --git`` boundaries,
       falling back to ``--- ``/ ``+++ `` boundaries for non-git diffs).
    2. Separates each section into a *header* (lines before the first ``@@``
       hunk) and *hunk body* (the actual changed lines).
    3. Includes **all** file headers first — they are short and tell the model
       which files changed.
    4. Distributes the remaining token budget evenly across every file's hunk
       body, truncating each one proportionally if needed.

    The result is always within *max_tokens* and always mentions every file
    that was touched, even if the per-file hunk content is abbreviated.
    """
    # Fast path: already fits.
    if len(enc.encode(diff_text, disallowed_special=())) <= max_tokens:
        return diff_text

    lines = diff_text.splitlines(keepends=True)

    # ── 1. Split into per-file sections ──────────────────────────────────────
    file_sections: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("diff --git ") and current:
            file_sections.append(current)
            current = []
        current.append(line)
    if current:
        file_sections.append(current)

    # If the diff has no "diff --git" markers, treat the whole thing as one
    # section and fall back to a plain token-level truncation.
    if len(file_sections) == 1 and not diff_text.lstrip().startswith("diff --git "):
        return enc.decode(enc.encode(diff_text, disallowed_special=())[:max_tokens])

    # ── 2. Separate header from hunk body for each file ──────────────────────
    # header = everything before the first "@@ " line
    # body   = the "@@ " line and everything after it within the same file
    file_parts: list[tuple[str, str]] = []
    for section in file_sections:
        header_lines: list[str] = []
        body_lines: list[str] = []
        in_body = False
        for line in section:
            if not in_body and line.startswith("@@ "):
                in_body = True
            if in_body:
                body_lines.append(line)
            else:
                header_lines.append(line)
        file_parts.append(("".join(header_lines), "".join(body_lines)))

    # ── 3. Tokenise headers and bodies ───────────────────────────────────────
    header_token_lists = [enc.encode(h, disallowed_special=()) for h, _ in file_parts]
    body_token_lists   = [enc.encode(b, disallowed_special=()) for _, b in file_parts]

    total_header_tokens = sum(len(t) for t in header_token_lists)

    # Edge case: headers alone exceed the budget — include as many whole
    # file headers as possible and stop.
    if total_header_tokens >= max_tokens:
        result: list[int] = []
        for h_toks in header_token_lists:
            if len(result) + len(h_toks) > max_tokens:
                break
            result.extend(h_toks)
        return enc.decode(result)

    # ── 4. Distribute remaining budget evenly across hunk bodies ─────────────
    remaining = max_tokens - total_header_tokens
    n_files   = len(file_parts)
    per_file_budget = remaining // n_files  # integer floor is fine

    parts: list[str] = []
    for (header, _), h_toks, b_toks in zip(file_parts, header_token_lists, body_token_lists):
        parts.append(header)
        if b_toks:
            parts.append(enc.decode(b_toks[:per_file_budget]))

    return "".join(parts)


def format_prompt(diff: str, message: str = "") -> str:
    text = f"{DIFF_START}\n{diff}\n{DIFF_END}\n{MSG_START}\n"
    if message:
        text += f"{message}\n{MSG_END}"
    return text


def format_prompt_only(diff: str) -> str:
    return f"{DIFF_START}\n{diff}\n{DIFF_END}\n{MSG_START}\n"


class DiffDataset(Dataset):
    def __init__(self, entries: List[dict]):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


def collate_diff_batch(batch, enc, max_diff_tokens, max_msg_tokens,
                       allowed_max_length=1024, device="cpu"):
    input_ids_list = []
    prompt_lens = []

    for entry in batch:
        truncated_diff = truncate_diff(entry["diff"], enc, max_diff_tokens)

        prompt = format_prompt_only(truncated_diff)
        prompt_ids = enc.encode(prompt, allowed_special=_SPECIAL_TOKENS_SET)

        msg_ids = enc.encode(entry["message"], disallowed_special=())[:max_msg_tokens]
        end_ids = enc.encode("\n" + MSG_END, allowed_special=_SPECIAL_TOKENS_SET)

        full_ids = prompt_ids + msg_ids + end_ids
        full_ids = full_ids[:allowed_max_length - 1] + [PAD_ID]

        prompt_len = min(len(prompt_ids), len(full_ids) - 1)

        input_ids_list.append(full_ids)
        prompt_lens.append(prompt_len)

    batch_max = max(len(ids) for ids in input_ids_list)

    padded_inputs = []
    padded_targets = []
    padded_masks = []

    for ids, p_len in zip(input_ids_list, prompt_lens):
        real_len = len(ids)
        pad_amount = batch_max - real_len
        ids = ids + [PAD_ID] * pad_amount

        inputs = torch.tensor(ids[:-1], dtype=torch.long)
        targets = torch.tensor(ids[1:], dtype=torch.long)

        mask = [1] * (real_len - 1) + [0] * pad_amount
        attn_mask = torch.tensor(mask, dtype=torch.long)

        pad_positions = (targets == PAD_ID).nonzero(as_tuple=False).squeeze(-1)
        if pad_positions.numel() > 1:
            targets[pad_positions[1:]] = IGNORE_INDEX

        prompt_target_len = max(p_len - 1, 0)
        targets[:prompt_target_len] = IGNORE_INDEX

        padded_inputs.append(inputs)
        padded_targets.append(targets)
        padded_masks.append(attn_mask)

    input_ids = torch.stack(padded_inputs).to(device)
    target_ids = torch.stack(padded_targets).to(device)
    attn_masks = torch.stack(padded_masks).to(device)
    prompt_lens_t = torch.tensor(prompt_lens, dtype=torch.long, device=device)

    return input_ids, target_ids, attn_masks, prompt_lens_t


def build_loaders(train_data, val_data, test_data, enc, cfg: DataConfig,
                  batch_size=8, device="cpu"):
    collate = partial(collate_diff_batch, enc=enc,
                      max_diff_tokens=cfg.max_diff_tokens,
                      max_msg_tokens=cfg.max_msg_tokens,
                      allowed_max_length=cfg.context_length,
                      device=device)

    train_ds = DiffDataset(train_data)
    val_ds = DiffDataset(val_data)
    test_ds = DiffDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate, num_workers=0)

    return train_loader, val_loader, test_loader
