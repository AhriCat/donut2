# Donut Codebase Review

## Critical Issues

### 1. `logic.py` — Missing imports
**Lines 1-3**: No imports at all. Needs `torch` and `nn`.

```python
# Missing:
import torch
import torch.nn as nn
```

### 2. `transform.py` — Missing function `_closest_factor_pair`
**Line 18**: Calls `_closest_factor_pair(dim)` but this function is not defined or imported.
The function exists in `attn.py` as `_closest_factor_pair_int` but is not exported.

**Fix**: Either import from attn.py or duplicate the function:
```python
def _closest_factor_pair(n: int) -> Tuple[int, int]:
    import math
    root = int(math.sqrt(n))
    for delta in range(0, root + 1):
        b = root + delta
        if b > 0 and n % b == 0:
            return (b, n // b)
        b = root - delta
        if b > 0 and n % b == 0:
            return (b, n // b)
    return (1, n)
```

### 3. `train.py` — Multiple issues
**Line 1**: Malformed — import statement merged with comment:
```python
# BASIC TRAINING EXAMPLEfrom datasets import load_dataset  # WRONG
```
Should be:
```python
# BASIC TRAINING EXAMPLE
from datasets import load_dataset
```

**Line 14**: References undefined `model` variable before it's created:
```python
if not model:  # 'model' doesn't exist yet
```
Should be:
```python
model = None
tokenizer = None

if model is None:
```

**Line 15**: References undefined `tok` variable:
```python
tokenizer = tok  # 'tok' is never defined
```

**Line 11**: Redundant — already limited to 10k in line 10:
```python
ds_full = load_dataset("...", split="train[:10000]")
ds = ds_full.select(range(10_000))  # Redundant
```

---

## Medium Issues

### 4. `cycloidpos.py` — Unused imports
**Lines 5-7**: `F`, `spectral_norm`, `Optional`, `Tuple` are imported but never used.

### 5. `attn.py` — Unused imports
**Line 9**: `Optional`, `Tuple` imported but `Optional` is unused.

### 6. `model.py` — Unused imports
**Lines 1, 5**: `math` and `spectral_norm` are imported but never used.

### 7. `hymba.py` — Unused import
**Line 2**: `math` is imported but never used.

### 8. `hymba.py` — Dropout never applied
**Line 17**: `self.dropout = nn.Dropout(dropout)` is created but never used in forward().

### 9. `model.py` — Weight tying silent failure
**Lines 35-38**: The try/except silently swallows errors. Should at least warn:
```python
try:
    self.out.weight = self.embed.weight
except Exception as e:
    import warnings
    warnings.warn(f"Weight tying failed: {e}")
```

---

## Minor Issues / Suggestions

### 10. `attn.py` — `groups` parameter unused
**Line 34**: `self.groups = groups` is stored but never used in forward pass.
Either implement grouped attention or remove the parameter.

### 11. `hymba.py` — Sequential loop is slow
**Lines 46-55**: The for-loop over sequence length is not parallelizable.
This is a known limitation of recurrent architectures, but worth noting for performance.
Consider adding a comment or providing a parallel scan alternative for training.

### 12. `tokenizer.py` — Potential issue with representation caching
**Lines 320-331**: The uncertainty damping logic modifies `vec` in place after computing it,
but before caching. This is correct, but the flow is non-obvious. Consider restructuring.

### 13. `model.py` — No causal masking in attention
The model appears to be a decoder (autoregressive generation) but `FocusedAttentionGroup`
doesn't apply a causal mask. For training, this would allow the model to "see the future."

**Fix**: Add causal mask in attention:
```python
# In FocusedAttentionGroup.forward(), after computing logits:
causal_mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
```

---

## Summary

| File | Critical | Medium | Minor |
|------|----------|--------|-------|
| logic.py | 1 | 0 | 0 |
| transform.py | 1 | 0 | 0 |
| train.py | 1 | 0 | 0 |
| cycloidpos.py | 0 | 1 | 0 |
| attn.py | 0 | 1 | 1 |
| model.py | 0 | 1 | 1 |
| hymba.py | 0 | 2 | 1 |
| tokenizer.py | 0 | 0 | 1 |
| mhc.py | 0 | 0 | 0 |
| model_mhc.py | 0 | 0 | 0 |

**Total: 3 critical, 5 medium, 4 minor**

The critical issues will cause immediate crashes. The medium issues are functional bugs
or dead code. The minor issues are style/performance suggestions.
