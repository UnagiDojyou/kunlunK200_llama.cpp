#!/usr/bin/env python3
"""Compare greedy token IDs and full-vocabulary logits from llm-smoke."""
import array
import math
from pathlib import Path
import re
import sys

def read(prefix):
    values = array.array('f')
    data = Path(prefix + '.bin').read_bytes()
    values.frombytes(data)
    match = re.search(r'TOKENS mode=\w+((?: \d+)+)', Path(prefix + '.txt').read_text())
    if not match or not values or not all(map(math.isfinite, values)):
        raise RuntimeError('invalid inference result: ' + prefix)
    return values, list(map(int, match[1].split()))

reference, reference_tokens = read(sys.argv[1])
actual, actual_tokens = read(sys.argv[2])
if len(reference) != len(actual):
    raise RuntimeError('different logits lengths')
errors = [float(a) - float(b) for a, b in zip(actual, reference)]
nmse = sum(e*e for e in errors) / max(sum(float(v)*v for v in reference), 1e-30)
max_error = max(map(abs, errors))
same = reference_tokens == actual_tokens
print(f'LOGITS elements={len(actual)} max_abs={max_error:.8g} nmse={nmse:.8g} greedy_tokens_equal={same}')
print(f'CPU tokens={reference_tokens}; XPU tokens={actual_tokens}')
if '--teacher-forced' in sys.argv[3:]:
    nv = len(reference) // len(reference_tokens)
    kl = []
    delta_nll = []
    for i, target in enumerate(reference_tokens):
        r = reference[i*nv:(i+1)*nv]
        x = actual[i*nv:(i+1)*nv]
        mr, mx = max(r), max(x)
        lr = mr + math.log(sum(math.exp(v-mr) for v in r))
        lx = mx + math.log(sum(math.exp(v-mx) for v in x))
        kl.append(sum(math.exp(a-lr)*((a-lr)-(b-lx)) for a,b in zip(r,x)))
        delta_nll.append((lx-x[target])-(lr-r[target]))
    mean_kl = sum(kl)/len(kl)
    print(f'TEACHER_FORCED mean_KL={mean_kl:.8g} max_KL={max(kl):.8g} mean_target_NLL_delta={sum(delta_nll)/len(delta_nll):.8g}')
    if nmse > .01 or mean_kl > .05:
        sys.exit(1)
elif not same or nmse > 1e-3:
    sys.exit(1)
