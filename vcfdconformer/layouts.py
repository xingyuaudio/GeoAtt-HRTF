"""Canonical LAP/SONICOM context layouts."""
from __future__ import annotations

import numpy as np


CANONICAL_LAYOUTS = {
    3: np.asarray([4, 11, 414], dtype=np.int64),
    5: np.asarray([0, 4, 8, 203, 612], dtype=np.int64),
    19: np.asarray([
        0, 4, 8, 11, 14, 18, 22, 265, 269, 273, 278, 282, 286,
        529, 533, 537, 542, 546, 550,
    ], dtype=np.int64),
    100: np.asarray([
        0, 8, 19, 25, 33, 38, 50, 57, 65, 67, 75, 84, 92, 103, 117,
        122, 130, 134, 142, 149, 159, 168, 176, 184, 195, 201, 209, 214,
        226, 233, 241, 243, 251, 260, 268, 279, 293, 298, 306, 310, 318,
        325, 335, 344, 352, 360, 371, 377, 385, 390, 402, 409, 417, 419,
        427, 436, 444, 455, 469, 474, 482, 486, 494, 501, 511, 520, 528,
        536, 547, 553, 561, 566, 578, 585, 593, 595, 603, 612, 620, 631,
        645, 650, 658, 662, 670, 677, 687, 696, 704, 712, 723, 729, 737,
        742, 754, 761, 769, 771, 779, 788,
    ], dtype=np.int64),
}


def canonical_layout(size: int) -> np.ndarray:
    try:
        return CANONICAL_LAYOUTS[size].copy()
    except KeyError as error:
        raise ValueError(f"canonical size must be one of {sorted(CANONICAL_LAYOUTS)}") from error

