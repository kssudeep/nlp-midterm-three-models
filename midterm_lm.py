"""Midterm project: bigram vs. LSTM vs. small transformer on next-character prediction.

All three models are trained on the same text, with the same train/validation split,
batch size, context length, number of optimiser steps and evaluation windows, so the
comparison is fair. Only the architecture (and a sensible learning rate) differs.

Usage:
    python midterm_lm.py                 # full run, ~3000 steps per model
    python midterm_lm.py --steps 300     # quick smoke test
    python midterm_lm.py --device cpu    # force a device (cuda / mps / cpu)
    python midterm_lm.py --steps 8000 --out_dir runs/8000_steps   # longer run, kept separate

Outputs (written next to this script, or to --out_dir):
    results.md              results table (fill in the quality/notes columns yourself)
    samples/<model>.txt     generated text from each model
    loss_curves.png         train/validation loss against training step
"""
import argparse
import math
import os
import time
import urllib.request

# Lets PyTorch fall back to CPU for any op not yet implemented on Apple's MPS backend.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
            "data/tinyshakespeare/input.txt")
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_text(path):
    """Return the dataset text, downloading Tiny Shakespeare if it is not on disk."""
    if not os.path.exists(path):
        print(f"Downloading dataset to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


class CharData:
    """Character-level tokenizer plus a 90/10 train/validation split."""

    def __init__(self, text, block_size, batch_size, device, train_frac=0.9):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(train_frac * len(data))
        self.splits = {"train": data[:n], "val": data[n:]}
        self.block_size, self.batch_size, self.device = block_size, batch_size, device

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def get_batch(self, split, generator=None):
        """Random (x, y) windows; y is x shifted one character to the right."""
        data = self.splits[split]
        ix = torch.randint(len(data) - self.block_size - 1, (self.batch_size,),
                           generator=generator)
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)


# --------------------------------------------------------------------------- #
# Model 1: bigram (uses ONLY the current character)
# --------------------------------------------------------------------------- #
class BigramLM(nn.Module):
    """A lookup table: row = current character, columns = logits for the next one."""

    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)  # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# --------------------------------------------------------------------------- #
# Model 2: LSTM (non-transformer; carries a hidden state over the whole window)
# --------------------------------------------------------------------------- #
class LSTMLM(nn.Module):
    """Embedding -> 2-layer LSTM -> linear head. No attention."""

    def __init__(self, vocab_size, n_embd=128, hidden=256, n_layer=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(n_embd, hidden, num_layers=n_layer,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, idx, targets=None):
        out, _ = self.lstm(self.emb(idx))  # (B, T, hidden)
        logits = self.head(out)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# --------------------------------------------------------------------------- #
# Model 3: small transformer (decoder-only, GPT style)
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t may only look at positions <= t."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)   # queries, keys, values in one matmul
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))   # (B, n_head, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    """Pre-norm transformer block: attention then feedforward, each with a residual."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TinyTransformer(nn.Module):
    """Token embeddings + positional embeddings + N blocks + output layer."""

    def __init__(self, vocab_size, block_size, n_embd=128, n_head=4, n_layer=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)      # token embeddings
        self.pos_emb = nn.Embedding(block_size, n_embd)      # positional information
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout)
                                      for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)            # output layer

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# --------------------------------------------------------------------------- #
# Training, evaluation, generation (identical for all three models)
# --------------------------------------------------------------------------- #
def sync(device):
    """Wait for queued GPU work so wall-clock timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def estimate_loss(model, data, eval_iters):
    """Mean loss on fixed windows. The generator is re-seeded on every call, so every
    model and every evaluation sees exactly the same train/val windows."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        g = torch.Generator().manual_seed(0)
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = data.get_batch(split, generator=g)
            losses[i] = model(x, y)[1].item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_model(name, model, lr, data, args):
    """Train for a fixed number of steps; return the loss history and training time.
    Time counts optimiser steps only (evaluation is excluded)."""
    device = data.device
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name} | {n_params:,} parameters | lr={lr} ===")
    history, train_time = [], 0.0
    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(model, data, args.eval_iters)
            history.append((step, losses["train"], losses["val"]))
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}"
                  f" | train time {train_time:6.1f}s")
        if step == args.steps:
            break
        sync(device)
        t0 = time.time()
        xb, yb = data.get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sync(device)
        train_time += time.time() - t0
    return {"history": history, "time": train_time, "params": n_params}


@torch.no_grad()
def generate(model, data, n_chars, temperature=1.0):
    """Sample text one character at a time, feeding back at most block_size characters."""
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long, device=data.device)  # start from token 0
    for _ in range(n_chars):
        logits, _ = model(idx[:, -data.block_size:])
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    model.train()
    return data.decode(idx[0].tolist())


def pick_device(requested):
    """Choose cuda > mps > cpu unless the user asked for something specific."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_outputs(results, samples, args):
    """Write results.md, samples/*.txt and loss_curves.png to args.out_dir."""
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    lines =["| Model | Params | Train Loss | Validation Loss | Training Time (s) "
             "| Generated Text Quality | Notes |",
             "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        _, tr, va = r["history"][-1]
        lines.append(f"| {name} | {r['params']:,} | {tr:.4f} | {va:.4f} | {r['time']:.1f} "
                     f"| TODO (Low/Medium/High) | TODO |")
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(os.path.join(out, "samples"), exist_ok=True)
    for name, text in samples.items():
        fname = name.lower().replace(" ", "_").replace("-", "_") + ".txt"
        with open(os.path.join(out, "samples", fname), "w", encoding="utf-8") as f:
            f.write(text)
    print("\n" + "\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, r in results.items():
            steps, tr, va = zip(*r["history"])
            line, = ax.plot(steps, va, label=f"{name} (val)")
            ax.plot(steps, tr, "--", color=line.get_color(), alpha=0.6, label=f"{name} (train)")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Cross-entropy loss (nats)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "loss_curves.png"), dpi=150)
    except ImportError:
        print("matplotlib not installed; skipping loss_curves.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(HERE, "input.txt"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--sample_chars", type=int, default=600)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out_dir", default=HERE,
                        help="where to write results.md, samples/ and loss_curves.png")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Using device: {device}")
    data = CharData(load_text(args.data), args.block_size, args.batch_size, device)
    V = data.vocab_size
    print(f"Characters: {sum(len(s) for s in data.splits.values()):,} | vocab size: {V}")

    # (name, constructor, learning rate). Learning rates differ per architecture
    # (a lookup table tolerates a much larger one than a transformer).
    specs = [
        ("Bigram", lambda: BigramLM(V), 1e-2),
        ("LSTM", lambda: LSTMLM(V), 3e-3),
        ("Transformer", lambda: TinyTransformer(V, args.block_size), 1e-3),
    ]
    results, samples = {}, {}
    for name, build, lr in specs:
        torch.manual_seed(args.seed)          # same seed before each model is built
        model = build()
        results[name] = train_model(name, model, lr, data, args)
        torch.manual_seed(args.seed)          # same sampling randomness for each model
        samples[name] = generate(model, data, args.sample_chars)
        print(f"\n--- {name} sample ---\n{samples[name]}\n")
    save_outputs(results, samples, args)


if __name__ == "__main__":
    main()
