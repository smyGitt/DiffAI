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
                       allowed_max_length=512, device="cpu"):
    input_ids_list = []
    prompt_lens = []

    for entry in batch:
        diff_ids = enc.encode(entry["diff"])[:max_diff_tokens]
        truncated_diff = enc.decode(diff_ids)

        prompt = format_prompt_only(truncated_diff)
        prompt_ids = enc.encode(prompt)

        msg_ids = enc.encode(entry["message"])[:max_msg_tokens]
        end_ids = enc.encode("\n" + MSG_END)

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
