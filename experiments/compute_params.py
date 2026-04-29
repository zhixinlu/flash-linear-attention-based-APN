#!/usr/bin/env python3
"""Compute parameter counts for H=4 multi-head experiments."""

def count_params_transformer(D, H=4, L=10, ffn_mult=4):
    # per layer = QKV (3D^2) + O (D^2) + FFN (8D^2) + norms (4D)
    per_layer = 3*D*D + D*D + ffn_mult*D*D + ffn_mult*D*D + 4*D
    total = 3*D + D + L*per_layer + 2*D + D*10 + 10
    return total

def count_params_deltanet(D, H=4, L=10, use_ffn=False, ffn_mult=4):
    # per layer = Q,K,V (3D^2) + beta (D*H) + norm (2D) [+ FFN 8D^2 + norm 2D]
    per_layer = 3*D*D + D*H + 2*D
    if use_ffn:
        per_layer += ffn_mult*D*D + ffn_mult*D*D + 2*D
    total = 3*D + D + L*per_layer + 2*D + D*10 + 10
    return total

def count_params_apn(D, H=4, L=10, use_ffn=False, ffn_mult=4):
    # per layer = W (D^2) + norm (2D) + eta (H) + lam (H) [+ FFN 8D^2 + norm 2D]
    per_layer = D*D + 2*D + H + H
    if use_ffn:
        per_layer += ffn_mult*D*D + ffn_mult*D*D + 2*D
    total = 3*D + D + L*per_layer + 2*D + D*10 + 10
    return total

print(f'Job 1: Transformer D=100, H=4: {count_params_transformer(100):,}')

# Job 2: DeltaNet+FFN
for D in range(96, 120):
    if D % 4 == 0:
        p = count_params_deltanet(D, H=4, use_ffn=True)
        if abs(p - 1205610) < 30000:
            print(f'Job 2 candidate: DeltaNet+FFN D={D}, H=4: {p:,}')

# Job 3: APN+FFN
for D in range(100, 128):
    if D % 4 == 0:
        p = count_params_apn(D, H=4, use_ffn=True)
        if abs(p - 1205610) < 30000:
            print(f'Job 3 candidate: APN+FFN D={D}, H=4: {p:,}')

# Job 4: APN no FFN D=100 H=4
print(f'Job 4: APN D=100, H=4 (no FFN): {count_params_apn(100, H=4, use_ffn=False):,}')
