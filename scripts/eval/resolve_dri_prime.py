#!/usr/bin/env python3
"""Map HIP device order to Mesa's PCI-form DRI_PRIME selectors."""

from __future__ import annotations

import argparse
import ctypes
import os
import re
from pathlib import Path


PCI_ADDRESS = re.compile(r"([0-9a-fA-F]{4}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])")
VISIBILITY_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


def render_addresses() -> set[str]:
    addresses = set()
    for node in Path("/sys/class/drm").glob("renderD*"):
        if (Path("/dev/dri") / node.name).exists():
            addresses.add((node / "device").resolve().name.lower())
    return addresses


def selectors() -> list[str]:
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipGetDeviceCount.restype = ctypes.c_int
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    hip.hipDeviceGetPCIBusId.restype = ctypes.c_int

    count = ctypes.c_int()
    code = hip.hipGetDeviceCount(ctypes.byref(count))
    if code != 0 or count.value < 1:
        raise RuntimeError(f"HIP device discovery failed (status={code}, count={count.value})")

    available_render = render_addresses()
    if not available_render:
        raise RuntimeError("No DRM render nodes found; expose /dev/dri to the container")

    result = []
    for device_id in range(count.value):
        buffer = ctypes.create_string_buffer(64)
        code = hip.hipDeviceGetPCIBusId(buffer, len(buffer), device_id)
        if code != 0:
            raise RuntimeError(f"Cannot read PCI address for HIP device {device_id}: {code}")
        address = buffer.value.decode("ascii").lower()
        match = PCI_ADDRESS.fullmatch(address)
        if match is None:
            raise RuntimeError(f"Unexpected PCI address for HIP device {device_id}: {address!r}")
        if address not in available_render:
            raise RuntimeError(f"HIP device {device_id} ({address}) has no visible DRM render node")
        result.append("pci-" + "_".join(match.groups()) + "!")

    if len(set(result)) != len(result):
        raise RuntimeError("Multiple HIP devices resolved to the same PCI address")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="List all physical HIP devices in HIP order")
    group.add_argument("--visible-one", action="store_true", help="Resolve one already selected HIP device")
    args = parser.parse_args()

    if args.all:
        for name in VISIBILITY_VARS:
            os.environ.pop(name, None)

    found = selectors()
    if args.visible_one:
        if len(found) != 1:
            raise RuntimeError(f"Expected one visible HIP device, found {len(found)}")
        print(found[0])
    else:
        print(",".join(found))


if __name__ == "__main__":
    main()
