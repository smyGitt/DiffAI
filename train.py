import csv
import math
import random
import time

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from config import ModelConfig, DataConfig, TrainConfig
from data import load_commitbench, build_loaders, make_tokenizer

IGNORE_INDEX = -100
RESULTS_DIR = Path("results")


def build_run_name(tcfg: TrainConfig):
    parts = ["diffai"]
    parts.append(f"{int(tcfg.blind_rate * 100)}blind")
    parts.append(f"lr{tcfg.lr}")
    parts.append(f"bs{tcfg.batch_size}")
    parts.append(f"{tcfg.epochs}ep")
    if tcfg.freeze_backbone:
        parts.append(f"freeze{tcfg.unfreeze_last_n}")
    return "_".join(parts)


def calc_loss_batch(model, input_ids, target_ids, attn_mask, use_pretrained=False):
    device_type = input_ids.device.type
    with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
        if use_pretrained:
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        else:
            logits = model(input_ids)

    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.view(B * T, V),
        target_ids.view(B * T),
        ignore_index=IGNORE_INDEX
    )
    return loss


@torch.no_grad()
def eval_loss(model, loader, max_batches=50, use_pretrained=False):
    model.eval()
    losses = []
    for i, (input_ids, target_ids, attn_mask, _) in enumerate(loader):
        if i >= max_batches:
            break
        loss = calc_loss_batch(model, input_ids, target_ids, attn_mask, use_pretrained)
        losses.append(float(loss))
    model.train()
    return sum(losses) / max(len(losses), 1)


def freeze_backbone(model, unfreeze_last_n=4):
    for p in model.parameters():
        p.requires_grad = False

    for p in model.lm_head.parameters():
        p.requires_grad = True
    for p in model.transformer.ln_f.parameters():
        p.requires_grad = True
    for block in model.transformer.h[-unfreeze_last_n:]:
        for p in block.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")


def calc_perplexity_from_loss(loss):
    return math.exp(loss)


def train(model, train_loader, val_loader, tcfg: TrainConfig, run_name: str,
          use_pretrained=False):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tcfg.lr, weight_decay=0.01)

    total_steps = int(len(train_loader) * (1.0 - tcfg.blind_rate)) * tcfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{run_name}.csv"

    history = {"step": [], "train_loss": [], "val_loss": []}
    step = 0

    tcfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device_type = next(model.parameters()).device.type
    scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda"))

    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "epoch", "step", "train_loss", "train_perplexity",
            "val_loss", "val_perplexity", "lr", "batches_trained",
            "batches_skipped", "epoch_time_sec"
        ])

        model.train()
        for epoch in range(1, tcfg.epochs + 1):
            epoch_start = time.time()
            skipped = 0
            trained = 0
            n_batches = len(train_loader)

            for batch_idx, (input_ids, target_ids, attn_mask, _) in enumerate(train_loader):
                if random.random() < tcfg.blind_rate:
                    skipped += 1
                    pct = (batch_idx + 1) / n_batches * 100
                    print(f"\rEpoch {epoch} [{batch_idx + 1}/{n_batches}] {pct:5.1f}% | skipped", end="", flush=True)
                    continue

                step += 1
                trained += 1

                optimizer.zero_grad()
                loss = calc_loss_batch(model, input_ids, target_ids, attn_mask, use_pretrained)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                pct = (batch_idx + 1) / n_batches * 100
                print(f"\rEpoch {epoch} [{batch_idx + 1}/{n_batches}] {pct:5.1f}% | loss {float(loss):.4f}", end="", flush=True)

                if step % tcfg.eval_every == 0:
                    tr_loss = eval_loss(model, train_loader, max_batches=30, use_pretrained=use_pretrained)
                    va_loss = eval_loss(model, val_loader, max_batches=30, use_pretrained=use_pretrained)
                    history["step"].append(step)
                    history["train_loss"].append(tr_loss)
                    history["val_loss"].append(va_loss)
                    print(f"\n  eval step {step:6d} | train {tr_loss:.4f} (ppl {calc_perplexity_from_loss(tr_loss):.2f}) | val {va_loss:.4f} (ppl {calc_perplexity_from_loss(va_loss):.2f}) | lr {scheduler.get_last_lr()[0]:.2e}")

                if step % tcfg.save_every == 0:
                    path = tcfg.checkpoint_dir / f"diffai_step{step}.pt"
                    torch.save(model.state_dict(), path)
                    print(f"\n  saved {path}")

            epoch_time = time.time() - epoch_start
            print()

            tr_loss = eval_loss(model, train_loader, max_batches=50, use_pretrained=use_pretrained)
            va_loss = eval_loss(model, val_loader, max_batches=50, use_pretrained=use_pretrained)
            tr_ppl = calc_perplexity_from_loss(tr_loss)
            va_ppl = calc_perplexity_from_loss(va_loss)
            lr = scheduler.get_last_lr()[0]

            csv_writer.writerow([
                epoch, step, f"{tr_loss:.6f}", f"{tr_ppl:.4f}",
                f"{va_loss:.6f}", f"{va_ppl:.4f}", f"{lr:.2e}",
                trained, skipped, f"{epoch_time:.1f}"
            ])
            csv_file.flush()

            print(f"Epoch {epoch} done | train {tr_loss:.4f} (ppl {tr_ppl:.2f}) | val {va_loss:.4f} (ppl {va_ppl:.2f}) | {trained} batches, {skipped} skipped | {epoch_time:.0f}s")

    print(f"Saved epoch metrics to {csv_path}")

    final_path = tcfg.checkpoint_dir / "diffai_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved {final_path}")

    return history


def plot_history(history, run_name: str):
    if not history["step"]:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(history["step"], history["train_loss"], label="train")
    plt.plot(history["step"], history["val_loss"], label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(run_name)
    plt.legend()
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"{run_name}_loss.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Saved plot to {plot_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    random.seed(0)
    torch.manual_seed(0)

    mcfg = ModelConfig()
    dcfg = DataConfig()
    tcfg = TrainConfig()

    run_name = build_run_name(tcfg)
    print(f"Run: {run_name}")

    enc = make_tokenizer()

    print("Loading dataset...")
    train_data, val_data, test_data = load_commitbench(dcfg)

    print("Building dataloaders...")
    train_loader, val_loader, test_loader = build_loaders(
        train_data, val_data, test_data, enc, dcfg,
        batch_size=tcfg.batch_size, device=device
    )

    if tcfg.use_pretrained:
        print(f"Loading pretrained {tcfg.pretrained_model}...")
        from model import load_pretrained_gpt2
        model = load_pretrained_gpt2(mcfg, tcfg.pretrained_model).to(device)
        model.resize_token_embeddings(enc.n_vocab)
        if tcfg.freeze_backbone:
            freeze_backbone(model, tcfg.unfreeze_last_n)
        use_pretrained = True
    else:
        print("Training from scratch with DiffGPT...")
        from model import DiffGPT
        model = DiffGPT(mcfg).to(device)
        use_pretrained = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    tr0 = eval_loss(model, train_loader, max_batches=20, use_pretrained=use_pretrained)
    va0 = eval_loss(model, val_loader, max_batches=20, use_pretrained=use_pretrained)
    print(f"Before training | train {tr0:.4f} (ppl {calc_perplexity_from_loss(tr0):.2f}) | val {va0:.4f} (ppl {calc_perplexity_from_loss(va0):.2f})")

    history = train(model, train_loader, val_loader, tcfg, run_name, use_pretrained)
    plot_history(history, run_name)


if __name__ == "__main__":
    main()
