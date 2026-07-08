#!/usr/bin/env python3
"""Recreate the held-out eval sets.

The expert repos don't publish their standalone held-out files, but the
router repo's eval.py contains the canonical generator: random.seed(12345),
100 mult items ("What is {a} times {b}?", a,b in 10..99) then 100 roman items
("Convert {n} to Roman numerals", n in 1..3999). This script mirrors that
generation exactly (same seed, same draw order, same phrasing) so results are
comparable with the published end-to-end numbers, then extends mult with 100
extra items (seed 12346, disjoint stream) to reach the 200-item gate size.
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def int_to_roman(n: int) -> str:
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
            (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
            (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def main() -> None:
    # exact mirror of router repo eval.py
    rng = random.Random(12345)
    mult = []
    for _ in range(100):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        mult.append({"question": f"What is {a} times {b}?",
                     "answer": str(a * b), "domain": "mult"})
    roman = []
    for _ in range(100):
        n = rng.randint(1, 3999)
        roman.append({"question": f"Convert {n} to Roman numerals",
                      "answer": int_to_roman(n), "domain": "roman"})

    # extension to 200 mult items for the Step-3 gate (documented extra seed)
    rng2 = random.Random(12346)
    for _ in range(100):
        a, b = rng2.randint(10, 99), rng2.randint(10, 99)
        mult.append({"question": f"What is {a} times {b}?",
                     "answer": str(a * b), "domain": "mult"})

    for name, rows in (("mult_eval.jsonl", mult), ("roman_eval.jsonl", roman)):
        with open(HERE / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {HERE / name} ({len(rows)} items)")


if __name__ == "__main__":
    main()
