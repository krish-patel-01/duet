"""Compare two --logit-trace files step by step (plain Python, no numpy needed).

usage: python compare_traces.py ref.bin test.bin
Prints the worst max-abs logit difference, lowest cosine similarity, how often the top-1 token
agrees, and the mean/max KL divergence between the two softmax distributions.
"""
import math
import struct
import sys
from array import array


def load(path):
    with open(path, "rb") as f:
        nv = struct.unpack("<i", f.read(4))[0]
        data = array("f")
        data.frombytes(f.read())
    return nv, data


nv, a = load(sys.argv[1])
nv2, b = load(sys.argv[2])
assert nv == nv2, "different vocab sizes"
n = min(len(a), len(b)) // nv

worst_diff, cos_min, agree, kls, diffs = 0.0, 1.0, 0, [], []
for s in range(n):
    x = a[s * nv:(s + 1) * nv]
    y = b[s * nv:(s + 1) * nv]
    md = max(abs(p - q) for p, q in zip(x, y))
    dot = sum(p * q for p, q in zip(x, y))
    nx = math.sqrt(sum(p * p for p in x))
    ny = math.sqrt(sum(q * q for q in y))
    ax = max(range(nv), key=x.__getitem__)
    ay = max(range(nv), key=y.__getitem__)
    mx, my = max(x), max(y)
    lzx = mx + math.log(sum(math.exp(p - mx) for p in x))
    lzy = my + math.log(sum(math.exp(q - my) for q in y))
    kl = sum(math.exp(p - lzx) * ((p - lzx) - (q - lzy)) for p, q in zip(x, y))
    diffs.append(md)
    worst_diff = max(worst_diff, md)
    cos_min = min(cos_min, dot / (nx * ny))
    agree += ax == ay
    kls.append(kl)

diffs.sort()
print(f"steps {n}: max|diff| worst {worst_diff:.4f} median {diffs[n // 2]:.4f} | "
      f"cosine min {cos_min:.6f} | top-1 agree {100 * agree / n:.1f}% | "
      f"KL mean {sum(kls) / n:.2e} max {max(kls):.2e}")
