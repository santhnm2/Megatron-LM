# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference collective thresholds for the native (non batch-invariant) policy."""

KiB = 1024
MiB = 1024 * 1024
# Max size for each world size in case symmetric memory is available
# For different SM architectures
CUSTOM_ALL_REDUCE_MAX_SIZES = {
    "9.0": {
        2: 64 * MiB,  # 64 MB
        4: 32 * MiB,  # 32 MB
        6: MiB // 2,  # 512 KB
        8: MiB // 4,  # 256 KB
    },
    "10.0": {
        2: 2 * MiB,  # 2 MB
        4: 2 * MiB,  # 2 MB
        6: 1 * MiB,  # 1 MB
        8: 1 * MiB,  # 1 MB
    },
    "10.3": {
        2: 4 * MiB,  # 4 MB
        4: 4 * MiB,  # 4 MB
        6: 8 * MiB,  # 8 MB
        8: 4 * MiB,  # 4 MB
    },
}

SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    "9.0": {
        2: 64 * MiB,  # 64 MB
        4: 32 * MiB,  # 32 MB
        6: 64 * MiB,  # 64 MB
        8: 64 * MiB,  # 64 MB
    },
    "10.0": {
        2: 8 * MiB,  # 8 MB
        4: 32 * MiB,  # 32 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    },
    "10.3": {
        2: 4 * MiB,  # 4 MB
        4: 32 * MiB,  # 32 MB
        6: 32 * MiB,  # 32 MB
        8: 64 * MiB,  # 64 MB
    },
}
