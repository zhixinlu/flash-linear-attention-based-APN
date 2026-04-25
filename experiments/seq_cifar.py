#!/usr/bin/env python3
"""
Seq-CIFAR-10 benchmark: DeltaNet vs APN on sequential image classification.

Input:  CIFAR-10 images flattened to sequences of 1024 RGB pixels  [B, 1024, 3]
Output: 10-class classification from the last time-step            [B, 10]

Usage:
    # Train APN only (10 layers, 100 hidden, 50 epochs)
    python experiments/seq_cifar.py --model apn --mode fast --n-layers 10 --d-hidden 100 --epochs 50

    # Train DeltaNet only
    python experiments/seq_cifar.py --model deltanet --mode fast --n-layers 10 --d-hidden 100 --epochs 50

    # Compare both
    python experiments/seq_cifar.py --model both --mode fast --n-layers 10 --d-hidden 100 --epochs 50
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from fla.ops.delta_rule import fused_recurrent_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


# ---------------------------------------------------------------------------
# DeltaNet Layer
# ---------------------------------------------------------------------------

class DeltaNetLayer(nn.Module):
    """
    Single-head DeltaNet layer using the fused recurrent Triton kernel.

    Recurrence (delta rule):
        S_t = S_{t-1} + beta * k_t (v_t - S_{t-1} k_t)^T
        o_t = q_t^T S_t
    """

    def __init__(self, d_hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.k_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.v_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.b_proj = nn.Linear(d_hidden, 1, bias=False)
        self.o_norm = nn.LayerNorm(d_hidden)

    def forward(self, x):
        """x: [B, L, D] -> [B, L, D]"""
        q = self.q_proj(x)
        k = F.normalize(self.k_proj(x), p=2, dim=-1)
        v = F.silu(self.v_proj(x))
        beta = self.b_proj(x).sigmoid()                  # [B, L, 1]

        # fused_recurrent_delta_rule expects [B, T, H, D] with H=1
        o, _ = fused_recurrent_delta_rule(
            q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2),
            beta.squeeze(-1).unsqueeze(2),
        )
        return self.o_norm(o.squeeze(2))


# ---------------------------------------------------------------------------
# APN Layer (Associative Plasticity Network)
# ---------------------------------------------------------------------------

class APNLayer(nn.Module):
    """
    APN layer mapped onto the gated delta-rule chunk kernel.

    Original APN recurrence:
        x_act = tanh(x_t)
        h_t   = W @ x_act + M_{t-1} @ x_act
        M_t   = lam * M_{t-1} + eta * h_t @ x_act^T

    Mapped to gated delta-rule kernel:
        k = tanh(x)                    — key
        v = W @ tanh(x)                — value (static weight output)
        q = tanh(x)                    — query
        g = log(lam)                   — scalar decay
        beta = eta                     — learning rate on delta term
        + residual W @ tanh(x)         — static feedforward path

    Update: S_t = lam * S_{t-1} + eta * (k ⊗ (v - S @ k))
    eta is a free trainable scalar (can be positive or negative).
    lam is a trainable scalar (logit-space parameterization, always in (0, 1)).
    """

    def __init__(self, d_hidden: int, lam: float = 0.99, eta: float = 0.01):
        super().__init__()
        self.d = d_hidden
        self.eta = nn.Parameter(torch.tensor(float(eta)))
        # lam in (0,1) via sigmoid: logit(0.99) ≈ 4.595
        self.lam_logit = nn.Parameter(torch.tensor(lam).logit())
        self.W = nn.Linear(d_hidden, d_hidden, bias=False)
        self.o_norm = nn.LayerNorm(d_hidden)

    @property
    def lam(self):
        return self.lam_logit.sigmoid()

    def forward(self, x):
        """x: [B, L, D] -> [B, L, D]"""
        B, L, D = x.shape
        lam = self.lam

        x_act = torch.tanh(x)                                    # [B, L, D]
        static_out = self.W(x_act)                                # [B, L, D]

        # Kernel inputs [B, L, 1, D]
        k = x_act.unsqueeze(2)
        v = static_out.unsqueeze(2)
        q = x_act.unsqueeze(2)

        g = lam.log().expand(B, L, 1)                            # [B, L, 1]
        beta = self.eta.expand(B, L, 1)                           # [B, L, 1]

        o_dyn, _ = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=1.0,
            use_qk_l2norm_in_kernel=False,
        )

        return self.o_norm(static_out + o_dyn.squeeze(2))


# ---------------------------------------------------------------------------
# Sequential Model
# ---------------------------------------------------------------------------

class SeqModel(nn.Module):
    """
    Stack of recurrent layers for sequential classification.

    Architecture: input_proj → tanh → n_layers (with residual) → LayerNorm → classifier
    """

    def __init__(self, d_input: int, d_hidden: int, n_layers: int, n_classes: int,
                 layer_type: str = 'deltanet', apn_lam: float = 0.99, apn_eta: float = 0.01):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_hidden)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            if layer_type == 'deltanet':
                self.layers.append(DeltaNetLayer(d_hidden))
            elif layer_type == 'apn':
                self.layers.append(APNLayer(d_hidden, lam=apn_lam, eta=apn_eta))
            else:
                raise ValueError(f"Unknown layer_type: {layer_type}")
        self.norm = nn.LayerNorm(d_hidden)
        self.classifier = nn.Linear(d_hidden, n_classes)

    def forward(self, x):
        """x: [B, L, d_input] -> [B, n_classes]"""
        h = torch.tanh(self.input_proj(x))
        for layer in self.layers:
            h = h + layer(h)
        return self.classifier(self.norm(h[:, -1, :]))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_cifar10_loaders(batch_size: int, data_dir: str = './data', num_workers: int = 4):
    """CIFAR-10 images reshaped to [1024, 3] sequences (32×32 pixels, RGB features)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images).permute(0, 2, 3, 1).reshape(-1, 1024, 3)
        return images, torch.tensor(labels)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, epoch, log_interval=100):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [Epoch {epoch}] batch {batch_idx+1}/{len(loader)} | "
                  f"loss={total_loss/total:.4f} | acc={100.*correct/total:.2f}% | "
                  f"time={time.time()-t0:.1f}s")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Seq-CIFAR-10: DeltaNet vs APN")
    parser.add_argument('--model', type=str, default='both', choices=['deltanet', 'apn', 'both'])
    parser.add_argument('--d-hidden', type=int, default=100)
    parser.add_argument('--n-layers', type=int, default=10)
    parser.add_argument('--apn-lam', type=float, default=0.99, help='APN decay lambda')
    parser.add_argument('--apn-eta', type=float, default=0.01, help='APN learning rate eta (init)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log-interval', type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Device: {args.device}")
    print(f"Config: d_hidden={args.d_hidden}, n_layers={args.n_layers}")
    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}\n")

    train_loader, test_loader = get_cifar10_loaders(
        args.batch_size, data_dir=args.data_dir, num_workers=args.num_workers)

    models_to_run = []
    if args.model in ('deltanet', 'both'):
        models_to_run.append('deltanet')
    if args.model in ('apn', 'both'):
        models_to_run.append('apn')

    results = {}
    for model_name in models_to_run:
        print(f"{'='*60}\nTraining: {model_name.upper()}\n{'='*60}")

        model = SeqModel(
            d_input=3, d_hidden=args.d_hidden, n_layers=args.n_layers, n_classes=10,
            layer_type=model_name, apn_lam=args.apn_lam, apn_eta=args.apn_eta,
        ).to(args.device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_test_acc = 0.0
        history = []

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, args.device, epoch, args.log_interval)
            test_loss, test_acc = evaluate(model, test_loader, args.device)
            scheduler.step()
            elapsed = time.time() - t0
            best_test_acc = max(best_test_acc, test_acc)

            history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                                test_loss=test_loss, test_acc=test_acc, time=elapsed))

            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"train {train_loss:.4f} / {100*train_acc:.2f}% | "
                  f"test {test_loss:.4f} / {100*test_acc:.2f}% | "
                  f"best={100*best_test_acc:.2f}% | {elapsed:.1f}s")

            if epoch == 1 and args.device == 'cuda':
                print(f"  >> Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")

        peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if args.device == 'cuda' else 0.0
        total_time = sum(h['time'] for h in history)

        results[model_name] = dict(
            best_test_acc=best_test_acc, final_test_acc=test_acc, n_params=n_params,
            peak_mem_mb=peak_mb, total_time=total_time, avg_epoch_time=total_time / len(history))

        print(f"\n  Total: {total_time:.1f}s | Peak GPU: {peak_mb:.0f} MB\n")

    if len(results) > 1:
        print(f"\n{'='*60}\nCOMPARISON SUMMARY\n{'='*60}")
        for name, r in results.items():
            print(f"  {name.upper():12s} | params={r['n_params']:>8,} | "
                  f"best={100*r['best_test_acc']:.2f}% | "
                  f"mem={r['peak_mem_mb']:.0f}MB | "
                  f"avg_epoch={r['avg_epoch_time']:.1f}s")


if __name__ == '__main__':
    main()
