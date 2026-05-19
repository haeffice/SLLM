"""Host CPU / NUMA / memory probe.

Pure stdlib, no ``vllm`` / ``torch`` import — this must run on the AVX2-only
dev box (where the vLLM CPU wheel may not even import) as well as on the real
AMX Xeon target.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

log = logging.getLogger("be.topology")

# ISA flags that matter for the vLLM CPU backend.
ISA_FLAGS = (
    "avx2",
    "avx512f",
    "avx512_bf16",
    "avx512_fp16",
    "amx_int8",
    "amx_bf16",
)

_GIB = 1024 ** 3
_MIB = 1024 ** 2


@dataclass
class NumaNode:
    node_id: int
    cpus: list[int]
    ram_bytes: int


@dataclass
class HostInfo:
    mem_total: int
    mem_available: int
    numa_nodes: list[NumaNode]
    numa_source: str  # "numactl" | "sysfs" | "fallback-single-node"
    isa: dict[str, bool] = field(default_factory=dict)
    logical_cores: int = 0
    python_version: tuple[int, int] = (0, 0)
    gcc_version: tuple[int, int] | None = None

    @property
    def numa_count(self) -> int:
        return len(self.numa_nodes)

    @property
    def smallest_node_ram(self) -> int:
        return min((n.ram_bytes for n in self.numa_nodes), default=self.mem_total)


def _read_meminfo() -> tuple[int, int]:
    total = avail = 0
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
    except OSError as e:
        log.warning("cannot read /proc/meminfo: %s", e)
    return total, avail


def _parse_cpu_list(spec: str) -> list[int]:
    """Expand a Linux cpu-list like '0-3,8,10-11' into [0,1,2,3,8,10,11]."""
    cpus: list[int] = []
    for part in spec.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _numa_from_numactl() -> list[NumaNode] | None:
    try:
        out = subprocess.run(
            ["numactl", "-H"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError) as e:
        log.info("numactl -H unavailable: %s", e)
        return None

    cpus: dict[int, list[int]] = {}
    sizes: dict[int, int] = {}
    for line in out.splitlines():
        m = re.match(r"node (\d+) cpus:\s*(.*)", line)
        if m:
            ids = [int(x) for x in m.group(2).split()] if m.group(2).strip() else []
            cpus[int(m.group(1))] = ids
            continue
        m = re.match(r"node (\d+) size:\s*(\d+)\s*MB", line)
        if m:
            sizes[int(m.group(1))] = int(m.group(2)) * _MIB
    if not cpus:
        return None
    return [
        NumaNode(nid, cpus.get(nid, []), sizes.get(nid, 0))
        for nid in sorted(cpus)
    ]


def _numa_from_sysfs() -> list[NumaNode] | None:
    base = "/sys/devices/system/node"
    if not os.path.isdir(base):
        return None
    nodes: list[NumaNode] = []
    for name in sorted(os.listdir(base)):
        m = re.fullmatch(r"node(\d+)", name)
        if not m:
            continue
        nid = int(m.group(1))
        ndir = os.path.join(base, name)
        ram = 0
        try:
            with open(os.path.join(ndir, "meminfo")) as fh:
                for line in fh:
                    if "MemTotal:" in line:
                        ram = int(line.split()[3]) * 1024
                        break
        except OSError:
            pass
        cpus: list[int] = []
        try:
            with open(os.path.join(ndir, "cpulist")) as fh:
                cpus = _parse_cpu_list(fh.read())
        except OSError:
            pass
        nodes.append(NumaNode(nid, cpus, ram))
    return nodes or None


def _detect_isa() -> dict[str, bool]:
    flags: set[str] = set()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith(("flags", "Features")):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError as e:
        log.warning("cannot read /proc/cpuinfo: %s", e)
    return {flag: flag in flags for flag in ISA_FLAGS}


def _detect_gcc() -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            ["gcc", "-dumpfullversion", "-dumpversion"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        m = re.search(r"(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def probe_host() -> HostInfo:
    total, avail = _read_meminfo()

    nodes = _numa_from_numactl()
    source = "numactl"
    if not nodes:
        nodes = _numa_from_sysfs()
        source = "sysfs"
    if not nodes:
        nodes = [NumaNode(0, list(range(os.cpu_count() or 1)), total)]
        source = "fallback-single-node"
        log.warning("NUMA topology unknown — assuming a single node")

    info = HostInfo(
        mem_total=total,
        mem_available=avail,
        numa_nodes=nodes,
        numa_source=source,
        isa=_detect_isa(),
        logical_cores=os.cpu_count() or 0,
        python_version=sys.version_info[:2],
        gcc_version=_detect_gcc(),
    )
    log.info(
        "host — RAM %.1f GiB (avail %.1f GiB), %d NUMA node(s) via %s, "
        "%d logical cores, ISA %s",
        total / _GIB,
        avail / _GIB,
        info.numa_count,
        source,
        info.logical_cores,
        ",".join(k for k, v in info.isa.items() if v) or "none",
    )
    return info
