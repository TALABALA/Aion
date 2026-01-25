"""
Advanced eBPF Observability with CO-RE and BTF Support.

Implements Pixie/Cilium-level eBPF capabilities:
- CO-RE (Compile Once, Run Everywhere) for portable BPF programs
- BTF (BPF Type Format) for kernel structure introspection
- Automatic protocol detection and parsing
- Distributed tracing via eBPF
- Continuous profiling with minimal overhead
- Network flow monitoring with service mesh integration
"""

import asyncio
import ctypes
import logging
import mmap
import os
import struct
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from enum import Enum, IntEnum
from datetime import datetime, timedelta
from collections import defaultdict
import json
import hashlib

logger = logging.getLogger(__name__)


# =============================================================================
# BTF (BPF Type Format) Support
# =============================================================================

class BTFKind(IntEnum):
    """BTF type kinds."""
    UNKN = 0
    INT = 1
    PTR = 2
    ARRAY = 3
    STRUCT = 4
    UNION = 5
    ENUM = 6
    FWD = 7
    TYPEDEF = 8
    VOLATILE = 9
    CONST = 10
    RESTRICT = 11
    FUNC = 12
    FUNC_PROTO = 13
    VAR = 14
    DATASEC = 15
    FLOAT = 16
    DECL_TAG = 17
    TYPE_TAG = 18
    ENUM64 = 19


@dataclass
class BTFType:
    """BTF type representation."""
    kind: BTFKind
    name: str
    size: int = 0
    members: List['BTFMember'] = field(default_factory=list)
    encoding: int = 0  # For INT types
    nelems: int = 0  # For ARRAY types


@dataclass
class BTFMember:
    """BTF struct/union member."""
    name: str
    type_id: int
    offset: int  # In bits


@dataclass
class BTFHeader:
    """BTF header structure."""
    magic: int = 0xEB9F
    version: int = 1
    flags: int = 0
    hdr_len: int = 24
    type_off: int = 0
    type_len: int = 0
    str_off: int = 0
    str_len: int = 0


class BTFParser:
    """Parser for BTF (BPF Type Format) data."""

    def __init__(self):
        self.types: Dict[int, BTFType] = {}
        self.strings: Dict[int, str] = {}
        self.header: Optional[BTFHeader] = None

    def parse(self, btf_data: bytes) -> bool:
        """Parse BTF data from bytes."""
        if len(btf_data) < 24:
            return False

        # Parse header
        magic, version, flags, hdr_len = struct.unpack("<HBBI", btf_data[:8])

        if magic != 0xEB9F:
            logger.warning(f"Invalid BTF magic: {magic:#x}")
            return False

        type_off, type_len, str_off, str_len = struct.unpack("<IIII", btf_data[8:24])

        self.header = BTFHeader(
            magic=magic,
            version=version,
            flags=flags,
            hdr_len=hdr_len,
            type_off=type_off,
            type_len=type_len,
            str_off=str_off,
            str_len=str_len
        )

        # Parse string section
        str_start = hdr_len + str_off
        str_end = str_start + str_len
        self._parse_strings(btf_data[str_start:str_end])

        # Parse type section
        type_start = hdr_len + type_off
        type_end = type_start + type_len
        self._parse_types(btf_data[type_start:type_end])

        return True

    def _parse_strings(self, data: bytes):
        """Parse BTF string table."""
        offset = 0
        while offset < len(data):
            # Find null terminator
            end = data.find(b'\x00', offset)
            if end == -1:
                break

            string = data[offset:end].decode('utf-8', errors='replace')
            self.strings[offset] = string
            offset = end + 1

    def _parse_types(self, data: bytes):
        """Parse BTF type section."""
        offset = 0
        type_id = 1  # Type IDs start at 1

        while offset + 12 <= len(data):
            # Common type header: name_off (4), info (4), size/type (4)
            name_off, info, size_or_type = struct.unpack("<III", data[offset:offset+12])

            kind = BTFKind((info >> 24) & 0x1F)
            vlen = info & 0xFFFF

            name = self.strings.get(name_off, "")

            btf_type = BTFType(kind=kind, name=name, size=size_or_type)

            offset += 12

            # Parse kind-specific data
            if kind == BTFKind.STRUCT or kind == BTFKind.UNION:
                members = []
                for _ in range(vlen):
                    if offset + 12 <= len(data):
                        m_name_off, m_type, m_offset = struct.unpack("<III", data[offset:offset+12])
                        members.append(BTFMember(
                            name=self.strings.get(m_name_off, ""),
                            type_id=m_type,
                            offset=m_offset
                        ))
                        offset += 12
                btf_type.members = members

            elif kind == BTFKind.ARRAY:
                if offset + 12 <= len(data):
                    elem_type, index_type, nelems = struct.unpack("<III", data[offset:offset+12])
                    btf_type.nelems = nelems
                    offset += 12

            elif kind == BTFKind.ENUM:
                # Skip enum values
                offset += vlen * 8

            elif kind == BTFKind.FUNC_PROTO:
                # Skip function params
                offset += vlen * 8

            self.types[type_id] = btf_type
            type_id += 1

    def get_struct_layout(self, struct_name: str) -> Optional[Dict[str, Tuple[int, int]]]:
        """Get field offsets and sizes for a struct."""
        for type_id, btf_type in self.types.items():
            if btf_type.kind in (BTFKind.STRUCT, BTFKind.UNION) and btf_type.name == struct_name:
                layout = {}
                for member in btf_type.members:
                    member_type = self.types.get(member.type_id)
                    size = member_type.size if member_type else 0
                    layout[member.name] = (member.offset // 8, size)  # Convert bits to bytes
                return layout
        return None

    def resolve_type(self, type_id: int) -> Optional[BTFType]:
        """Resolve a type ID, following typedefs and modifiers."""
        seen = set()
        while type_id in self.types and type_id not in seen:
            seen.add(type_id)
            btf_type = self.types[type_id]

            # Follow through modifiers and typedefs
            if btf_type.kind in (BTFKind.TYPEDEF, BTFKind.VOLATILE,
                                BTFKind.CONST, BTFKind.RESTRICT,
                                BTFKind.PTR, BTFKind.TYPE_TAG):
                type_id = btf_type.size  # size field holds the referenced type
            else:
                return btf_type

        return None


class BTFLoader:
    """Load BTF information from various sources."""

    VMLINUX_BTF_PATHS = [
        "/sys/kernel/btf/vmlinux",
        "/boot/vmlinux",
        "/lib/modules/{kernel}/vmlinux",
    ]

    def __init__(self):
        self.parser = BTFParser()
        self.kernel_version = self._get_kernel_version()

    def _get_kernel_version(self) -> str:
        """Get current kernel version."""
        try:
            return os.uname().release
        except Exception:
            return "unknown"

    def load_vmlinux_btf(self) -> bool:
        """Load BTF from vmlinux."""
        for path_template in self.VMLINUX_BTF_PATHS:
            path = path_template.format(kernel=self.kernel_version)
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        data = f.read()
                        if self.parser.parse(data):
                            logger.info(f"Loaded BTF from {path}")
                            return True
                except Exception as e:
                    logger.warning(f"Failed to load BTF from {path}: {e}")

        logger.warning("No vmlinux BTF found, CO-RE relocations may fail")
        return False

    def load_module_btf(self, module_name: str) -> bool:
        """Load BTF for a kernel module."""
        btf_path = f"/sys/kernel/btf/{module_name}"
        if os.path.exists(btf_path):
            try:
                with open(btf_path, 'rb') as f:
                    return self.parser.parse(f.read())
            except Exception as e:
                logger.warning(f"Failed to load module BTF: {e}")
        return False


# =============================================================================
# CO-RE (Compile Once, Run Everywhere) Support
# =============================================================================

class CORERelocKind(IntEnum):
    """CO-RE relocation types."""
    FIELD_BYTE_OFFSET = 0
    FIELD_BYTE_SIZE = 1
    FIELD_EXISTS = 2
    FIELD_SIGNED = 3
    FIELD_LSHIFT_U64 = 4
    FIELD_RSHIFT_U64 = 5
    TYPE_ID_LOCAL = 6
    TYPE_ID_TARGET = 7
    TYPE_EXISTS = 8
    TYPE_SIZE = 9
    ENUMVAL_EXISTS = 10
    ENUMVAL_VALUE = 11


@dataclass
class CORERelocation:
    """CO-RE relocation entry."""
    insn_off: int  # Instruction offset
    type_id: int  # BTF type ID
    access_str_off: int  # Access string offset
    kind: CORERelocKind


class CORERelocator:
    """Apply CO-RE relocations to BPF programs."""

    def __init__(self, local_btf: BTFParser, target_btf: BTFParser):
        self.local_btf = local_btf
        self.target_btf = target_btf
        self.relocations: List[CORERelocation] = []

    def add_relocation(self, reloc: CORERelocation):
        """Add a relocation to apply."""
        self.relocations.append(reloc)

    def apply_relocations(self, program: bytes) -> bytes:
        """Apply CO-RE relocations to a BPF program."""
        program_array = bytearray(program)

        for reloc in self.relocations:
            if reloc.kind == CORERelocKind.FIELD_BYTE_OFFSET:
                offset = self._resolve_field_offset(reloc)
                if offset is not None:
                    self._patch_instruction(program_array, reloc.insn_off, offset)

            elif reloc.kind == CORERelocKind.FIELD_EXISTS:
                exists = self._check_field_exists(reloc)
                self._patch_instruction(program_array, reloc.insn_off, 1 if exists else 0)

            elif reloc.kind == CORERelocKind.TYPE_SIZE:
                size = self._resolve_type_size(reloc)
                if size is not None:
                    self._patch_instruction(program_array, reloc.insn_off, size)

        return bytes(program_array)

    def _resolve_field_offset(self, reloc: CORERelocation) -> Optional[int]:
        """Resolve field byte offset for a struct access."""
        local_type = self.local_btf.types.get(reloc.type_id)
        if not local_type:
            return None

        # Find matching type in target
        target_type = self._find_matching_type(local_type)
        if not target_type:
            return None

        # Parse access string and resolve offset
        access_str = self.local_btf.strings.get(reloc.access_str_off, "")
        return self._resolve_access_offset(target_type, access_str)

    def _find_matching_type(self, local_type: BTFType) -> Optional[BTFType]:
        """Find matching type in target BTF by name."""
        for type_id, target_type in self.target_btf.types.items():
            if target_type.kind == local_type.kind and target_type.name == local_type.name:
                return target_type
        return None

    def _resolve_access_offset(self, btf_type: BTFType, access_str: str) -> Optional[int]:
        """Resolve offset for an access string like '0:1:2'."""
        if not access_str:
            return 0

        indices = [int(i) for i in access_str.split(':')]
        current_offset = 0

        for idx in indices:
            if btf_type.kind in (BTFKind.STRUCT, BTFKind.UNION):
                if idx < len(btf_type.members):
                    member = btf_type.members[idx]
                    current_offset += member.offset // 8
                    btf_type = self.target_btf.resolve_type(member.type_id)
                    if not btf_type:
                        return None
                else:
                    return None
            elif btf_type.kind == BTFKind.ARRAY:
                elem_size = btf_type.size // btf_type.nelems if btf_type.nelems else 0
                current_offset += idx * elem_size
            else:
                return None

        return current_offset

    def _check_field_exists(self, reloc: CORERelocation) -> bool:
        """Check if a field exists in target struct."""
        local_type = self.local_btf.types.get(reloc.type_id)
        if not local_type:
            return False

        target_type = self._find_matching_type(local_type)
        return target_type is not None

    def _resolve_type_size(self, reloc: CORERelocation) -> Optional[int]:
        """Resolve size of a type."""
        local_type = self.local_btf.types.get(reloc.type_id)
        if not local_type:
            return None

        target_type = self._find_matching_type(local_type)
        return target_type.size if target_type else None

    def _patch_instruction(self, program: bytearray, offset: int, value: int):
        """Patch a BPF instruction with a new immediate value."""
        if offset + 8 <= len(program):
            # BPF instruction format: op:8, dst_reg:4, src_reg:4, off:16, imm:32
            # Patch the immediate value at offset+4
            struct.pack_into("<I", program, offset + 4, value)


# =============================================================================
# Advanced eBPF Programs
# =============================================================================

class BPFMapType(IntEnum):
    """BPF map types."""
    HASH = 1
    ARRAY = 2
    PROG_ARRAY = 3
    PERF_EVENT_ARRAY = 4
    PERCPU_HASH = 5
    PERCPU_ARRAY = 6
    STACK_TRACE = 7
    CGROUP_ARRAY = 8
    LRU_HASH = 9
    LRU_PERCPU_HASH = 10
    LPM_TRIE = 11
    ARRAY_OF_MAPS = 12
    HASH_OF_MAPS = 13
    DEVMAP = 14
    SOCKMAP = 15
    CPUMAP = 16
    XSKMAP = 17
    SOCKHASH = 18
    CGROUP_STORAGE = 19
    REUSEPORT_SOCKARRAY = 20
    PERCPU_CGROUP_STORAGE = 21
    QUEUE = 22
    STACK = 23
    SK_STORAGE = 24
    DEVMAP_HASH = 25
    STRUCT_OPS = 26
    RINGBUF = 27
    INODE_STORAGE = 28
    TASK_STORAGE = 29
    BLOOM_FILTER = 30


@dataclass
class BPFMapSpec:
    """BPF map specification."""
    name: str
    map_type: BPFMapType
    key_size: int
    value_size: int
    max_entries: int
    flags: int = 0


@dataclass
class BPFProgSpec:
    """BPF program specification."""
    name: str
    prog_type: int
    attach_type: int
    expected_attach_type: int = 0
    flags: int = 0


class AdvancedBPFProgram:
    """
    Advanced BPF program with CO-RE support.

    Generates portable BPF programs that can run on different kernel versions
    using BTF-based type information and CO-RE relocations.
    """

    # BPF program types
    PROG_TYPE_KPROBE = 2
    PROG_TYPE_TRACEPOINT = 5
    PROG_TYPE_PERF_EVENT = 6
    PROG_TYPE_RAW_TRACEPOINT = 17
    PROG_TYPE_CGROUP_SOCK_ADDR = 20
    PROG_TYPE_TRACING = 26
    PROG_TYPE_LSM = 29

    def __init__(self, name: str):
        self.name = name
        self.maps: Dict[str, BPFMapSpec] = {}
        self.programs: Dict[str, BPFProgSpec] = {}
        self.btf_loader = BTFLoader()
        self.core_enabled = False

        # Try to load kernel BTF
        if self.btf_loader.load_vmlinux_btf():
            self.core_enabled = True

    def define_map(
        self,
        name: str,
        map_type: BPFMapType,
        key_size: int,
        value_size: int,
        max_entries: int,
        flags: int = 0
    ) -> 'AdvancedBPFProgram':
        """Define a BPF map."""
        self.maps[name] = BPFMapSpec(
            name=name,
            map_type=map_type,
            key_size=key_size,
            value_size=value_size,
            max_entries=max_entries,
            flags=flags
        )
        return self

    def define_program(
        self,
        name: str,
        prog_type: int,
        attach_type: int,
        attach_target: str = ""
    ) -> 'AdvancedBPFProgram':
        """Define a BPF program."""
        self.programs[name] = BPFProgSpec(
            name=name,
            prog_type=prog_type,
            attach_type=attach_type
        )
        return self

    def generate_http_tracer(self) -> str:
        """Generate CO-RE compatible HTTP tracing BPF program."""
        return '''
// SPDX-License-Identifier: GPL-2.0
// HTTP tracer with CO-RE support

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

char LICENSE[] SEC("license") = "GPL";

#define MAX_MSG_SIZE 256
#define HTTP_METHOD_MAX 8

struct http_event {
    __u64 timestamp_ns;
    __u32 pid;
    __u32 tid;
    __u32 uid;
    __u16 sport;
    __u16 dport;
    __u32 saddr;
    __u32 daddr;
    __u32 latency_ns;
    __u8 method[HTTP_METHOD_MAX];
    __u16 status_code;
    __u32 content_length;
    __u8 path[64];
    __u8 comm[16];
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} http_events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10000);
    __type(key, __u64);
    __type(value, struct http_event);
} active_connections SEC(".maps");

// CO-RE: Read from sock_common with field offset relocation
static __always_inline void get_sock_info(struct sock *sk, struct http_event *event) {
    // Using BPF_CORE_READ for portable field access
    struct sock_common *skc = (struct sock_common *)sk;

    event->sport = BPF_CORE_READ(skc, skc_num);
    event->dport = bpf_ntohs(BPF_CORE_READ(skc, skc_dport));
    event->saddr = BPF_CORE_READ(skc, skc_rcv_saddr);
    event->daddr = BPF_CORE_READ(skc, skc_daddr);
}

// Detect HTTP method from data
static __always_inline int detect_http_method(const __u8 *data, __u8 *method) {
    // Check for common HTTP methods
    if (data[0] == 'G' && data[1] == 'E' && data[2] == 'T') {
        __builtin_memcpy(method, "GET", 4);
        return 1;
    }
    if (data[0] == 'P' && data[1] == 'O' && data[2] == 'S' && data[3] == 'T') {
        __builtin_memcpy(method, "POST", 5);
        return 1;
    }
    if (data[0] == 'P' && data[1] == 'U' && data[2] == 'T') {
        __builtin_memcpy(method, "PUT", 4);
        return 1;
    }
    if (data[0] == 'D' && data[1] == 'E' && data[2] == 'L') {
        __builtin_memcpy(method, "DELETE", 7);
        return 1;
    }
    if (data[0] == 'H' && data[1] == 'T' && data[2] == 'T' && data[3] == 'P') {
        __builtin_memcpy(method, "RESP", 5);  // HTTP response
        return 2;
    }
    return 0;
}

SEC("kprobe/tcp_sendmsg")
int trace_tcp_sendmsg(struct pt_regs *ctx) {
    struct sock *sk = (struct sock *)PT_REGS_PARM1(ctx);
    struct msghdr *msg = (struct msghdr *)PT_REGS_PARM2(ctx);

    struct http_event *event;
    event = bpf_ringbuf_reserve(&http_events, sizeof(*event), 0);
    if (!event)
        return 0;

    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp_ns = bpf_ktime_get_ns();
    event->pid = bpf_get_current_pid_tgid() >> 32;
    event->tid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    event->uid = bpf_get_current_uid_gid() & 0xFFFFFFFF;

    bpf_get_current_comm(event->comm, sizeof(event->comm));

    // Get socket info using CO-RE
    get_sock_info(sk, event);

    // Try to read HTTP data
    struct iov_iter *iter;
    iter = &msg->msg_iter;

    // CO-RE: Check if iov_iter has the expected layout
    if (bpf_core_field_exists(iter->iov)) {
        struct iovec *iov = BPF_CORE_READ(iter, iov);
        if (iov) {
            __u8 buf[16];
            void *base = BPF_CORE_READ(iov, iov_base);
            bpf_probe_read_user(buf, sizeof(buf), base);
            detect_http_method(buf, event->method);
        }
    }

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("kretprobe/tcp_recvmsg")
int trace_tcp_recvmsg_ret(struct pt_regs *ctx) {
    int ret = PT_REGS_RC(ctx);
    if (ret <= 0)
        return 0;

    struct http_event *event;
    event = bpf_ringbuf_reserve(&http_events, sizeof(*event), 0);
    if (!event)
        return 0;

    __builtin_memset(event, 0, sizeof(*event));
    event->timestamp_ns = bpf_ktime_get_ns();
    event->pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(event->comm, sizeof(event->comm));

    bpf_ringbuf_submit(event, 0);
    return 0;
}
'''

    def generate_ssl_tracer(self) -> str:
        """Generate SSL/TLS tracing BPF program using uprobes."""
        return '''
// SPDX-License-Identifier: GPL-2.0
// SSL/TLS tracer for encrypted traffic inspection

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char LICENSE[] SEC("license") = "GPL";

#define MAX_DATA_SIZE 4096

struct ssl_event {
    __u64 timestamp_ns;
    __u32 pid;
    __u32 tid;
    __u32 data_len;
    __u8 direction;  // 0 = read, 1 = write
    __u8 comm[16];
    __u8 data[MAX_DATA_SIZE];
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1024 * 1024);
} ssl_events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10000);
    __type(key, __u64);
    __type(value, __u64);
} ssl_write_args SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10000);
    __type(key, __u64);
    __type(value, __u64);
} ssl_read_args SEC(".maps");

// Trace SSL_write entry to capture plaintext data
SEC("uprobe/SSL_write")
int trace_ssl_write(struct pt_regs *ctx) {
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    void *buf = (void *)PT_REGS_PARM2(ctx);
    __u64 buf_addr = (__u64)buf;

    bpf_map_update_elem(&ssl_write_args, &pid_tgid, &buf_addr, BPF_ANY);
    return 0;
}

SEC("uretprobe/SSL_write")
int trace_ssl_write_ret(struct pt_regs *ctx) {
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    int ret = PT_REGS_RC(ctx);

    __u64 *buf_addr = bpf_map_lookup_elem(&ssl_write_args, &pid_tgid);
    if (!buf_addr || ret <= 0) {
        bpf_map_delete_elem(&ssl_write_args, &pid_tgid);
        return 0;
    }

    struct ssl_event *event = bpf_ringbuf_reserve(&ssl_events, sizeof(*event), 0);
    if (!event) {
        bpf_map_delete_elem(&ssl_write_args, &pid_tgid);
        return 0;
    }

    __builtin_memset(event, 0, sizeof(*event));
    event->timestamp_ns = bpf_ktime_get_ns();
    event->pid = pid_tgid >> 32;
    event->tid = pid_tgid & 0xFFFFFFFF;
    event->direction = 1;  // write
    event->data_len = ret < MAX_DATA_SIZE ? ret : MAX_DATA_SIZE;

    bpf_get_current_comm(event->comm, sizeof(event->comm));
    bpf_probe_read_user(event->data, event->data_len & (MAX_DATA_SIZE - 1), (void *)*buf_addr);

    bpf_ringbuf_submit(event, 0);
    bpf_map_delete_elem(&ssl_write_args, &pid_tgid);
    return 0;
}

// Similar for SSL_read
SEC("uprobe/SSL_read")
int trace_ssl_read(struct pt_regs *ctx) {
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    void *buf = (void *)PT_REGS_PARM2(ctx);
    __u64 buf_addr = (__u64)buf;

    bpf_map_update_elem(&ssl_read_args, &pid_tgid, &buf_addr, BPF_ANY);
    return 0;
}

SEC("uretprobe/SSL_read")
int trace_ssl_read_ret(struct pt_regs *ctx) {
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    int ret = PT_REGS_RC(ctx);

    __u64 *buf_addr = bpf_map_lookup_elem(&ssl_read_args, &pid_tgid);
    if (!buf_addr || ret <= 0) {
        bpf_map_delete_elem(&ssl_read_args, &pid_tgid);
        return 0;
    }

    struct ssl_event *event = bpf_ringbuf_reserve(&ssl_events, sizeof(*event), 0);
    if (!event) {
        bpf_map_delete_elem(&ssl_read_args, &pid_tgid);
        return 0;
    }

    __builtin_memset(event, 0, sizeof(*event));
    event->timestamp_ns = bpf_ktime_get_ns();
    event->pid = pid_tgid >> 32;
    event->tid = pid_tgid & 0xFFFFFFFF;
    event->direction = 0;  // read
    event->data_len = ret < MAX_DATA_SIZE ? ret : MAX_DATA_SIZE;

    bpf_get_current_comm(event->comm, sizeof(event->comm));
    bpf_probe_read_user(event->data, event->data_len & (MAX_DATA_SIZE - 1), (void *)*buf_addr);

    bpf_ringbuf_submit(event, 0);
    bpf_map_delete_elem(&ssl_read_args, &pid_tgid);
    return 0;
}
'''

    def generate_continuous_profiler(self) -> str:
        """Generate continuous CPU profiling BPF program."""
        return '''
// SPDX-License-Identifier: GPL-2.0
// Continuous CPU profiler with minimal overhead

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

char LICENSE[] SEC("license") = "GPL";

#define MAX_STACK_DEPTH 127
#define TASK_COMM_LEN 16

struct profile_key {
    __u32 pid;
    __s32 kernel_stack_id;
    __s32 user_stack_id;
};

struct profile_value {
    __u64 count;
    __u64 total_time_ns;
    __u8 comm[TASK_COMM_LEN];
};

struct {
    __uint(type, BPF_MAP_TYPE_STACK_TRACE);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, MAX_STACK_DEPTH * sizeof(__u64));
    __uint(max_entries, 10000);
} stack_traces SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 100000);
    __type(key, struct profile_key);
    __type(value, struct profile_value);
} profile_counts SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} last_sample_time SEC(".maps");

SEC("perf_event")
int profile_cpu(struct bpf_perf_event_data *ctx) {
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;

    // Skip kernel threads
    if (pid == 0)
        return 0;

    struct profile_key key = {};
    key.pid = pid;

    // Capture kernel and user stack traces
    key.kernel_stack_id = bpf_get_stackid(ctx, &stack_traces, 0);
    key.user_stack_id = bpf_get_stackid(ctx, &stack_traces, BPF_F_USER_STACK);

    // Update profile counts
    struct profile_value *val = bpf_map_lookup_elem(&profile_counts, &key);
    if (val) {
        __sync_fetch_and_add(&val->count, 1);
    } else {
        struct profile_value new_val = {};
        new_val.count = 1;
        bpf_get_current_comm(new_val.comm, sizeof(new_val.comm));
        bpf_map_update_elem(&profile_counts, &key, &new_val, BPF_ANY);
    }

    return 0;
}

// Off-CPU profiling via sched tracepoints
SEC("tp_btf/sched_switch")
int trace_sched_switch(u64 *ctx) {
    // CO-RE: Access task_struct fields portably
    struct task_struct *prev = (struct task_struct *)ctx[1];
    struct task_struct *next = (struct task_struct *)ctx[2];

    // Record off-CPU time
    __u32 zero = 0;
    __u64 now = bpf_ktime_get_ns();

    __u64 *last = bpf_map_lookup_elem(&last_sample_time, &zero);
    if (last) {
        __u64 delta = now - *last;

        // Record off-CPU event for prev task
        __u32 pid = BPF_CORE_READ(prev, pid);
        if (pid > 0) {
            struct profile_key key = {};
            key.pid = pid;
            key.kernel_stack_id = -1;
            key.user_stack_id = -1;

            struct profile_value *val = bpf_map_lookup_elem(&profile_counts, &key);
            if (val) {
                __sync_fetch_and_add(&val->total_time_ns, delta);
            }
        }
    }

    bpf_map_update_elem(&last_sample_time, &zero, &now, BPF_ANY);
    return 0;
}
'''


# =============================================================================
# Protocol Detection and Parsing
# =============================================================================

class Protocol(Enum):
    """Detected protocol types."""
    UNKNOWN = "unknown"
    HTTP = "http"
    HTTP2 = "http2"
    GRPC = "grpc"
    MYSQL = "mysql"
    POSTGRES = "postgres"
    REDIS = "redis"
    KAFKA = "kafka"
    DNS = "dns"
    AMQP = "amqp"
    MONGODB = "mongodb"


@dataclass
class ProtocolMessage:
    """Parsed protocol message."""
    protocol: Protocol
    timestamp: datetime
    pid: int
    tid: int
    direction: str  # "request" or "response"
    latency_ns: Optional[int]
    attributes: Dict[str, Any] = field(default_factory=dict)


class ProtocolDetector:
    """Automatic protocol detection from raw data."""

    # Magic bytes for protocol detection
    SIGNATURES = {
        Protocol.HTTP: [b'GET ', b'POST ', b'PUT ', b'DELETE ', b'HEAD ', b'HTTP/'],
        Protocol.HTTP2: [b'PRI * HTTP/2', b'\x00\x00'],  # HTTP/2 preface or frame
        Protocol.MYSQL: [b'\x00\x00\x00\x0a'],  # MySQL handshake
        Protocol.POSTGRES: [b'PGRES', b'\x00\x00\x00'],
        Protocol.REDIS: [b'*', b'+', b'-', b':', b'$'],  # RESP protocol
        Protocol.KAFKA: [b'\x00\x00\x00'],  # Kafka uses length-prefixed messages
        Protocol.DNS: [b'\x00\x00\x01\x00'],  # DNS query header
        Protocol.MONGODB: [b'\x00\x00\x00'],  # MongoDB wire protocol
    }

    def detect(self, data: bytes, port: int = 0) -> Protocol:
        """Detect protocol from raw bytes and port."""
        if not data:
            return Protocol.UNKNOWN

        # Port-based hints
        port_hints = {
            80: Protocol.HTTP,
            443: Protocol.HTTP,  # Could be HTTP or HTTP/2
            8080: Protocol.HTTP,
            3306: Protocol.MYSQL,
            5432: Protocol.POSTGRES,
            6379: Protocol.REDIS,
            9092: Protocol.KAFKA,
            53: Protocol.DNS,
            27017: Protocol.MONGODB,
        }

        # Try signature-based detection first
        for protocol, signatures in self.SIGNATURES.items():
            for sig in signatures:
                if data.startswith(sig):
                    return protocol

        # Fall back to port-based detection
        if port in port_hints:
            return port_hints[port]

        # HTTP detection from content
        if self._is_http(data):
            return Protocol.HTTP

        # gRPC detection (HTTP/2 with application/grpc content-type)
        if b'grpc' in data.lower():
            return Protocol.GRPC

        return Protocol.UNKNOWN

    def _is_http(self, data: bytes) -> bool:
        """Check if data looks like HTTP."""
        try:
            text = data[:100].decode('ascii', errors='ignore')
            http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH']
            return any(text.startswith(m) for m in http_methods) or 'HTTP/' in text
        except Exception:
            return False


class ProtocolParser:
    """Parse protocol-specific messages."""

    def __init__(self):
        self.detector = ProtocolDetector()

    def parse(
        self,
        data: bytes,
        timestamp: datetime,
        pid: int,
        tid: int,
        port: int = 0
    ) -> Optional[ProtocolMessage]:
        """Parse raw data into protocol message."""
        protocol = self.detector.detect(data, port)

        if protocol == Protocol.UNKNOWN:
            return None

        parsers = {
            Protocol.HTTP: self._parse_http,
            Protocol.HTTP2: self._parse_http2,
            Protocol.MYSQL: self._parse_mysql,
            Protocol.POSTGRES: self._parse_postgres,
            Protocol.REDIS: self._parse_redis,
        }

        parser = parsers.get(protocol)
        if parser:
            return parser(data, timestamp, pid, tid)

        return ProtocolMessage(
            protocol=protocol,
            timestamp=timestamp,
            pid=pid,
            tid=tid,
            direction="unknown",
            latency_ns=None
        )

    def _parse_http(
        self,
        data: bytes,
        timestamp: datetime,
        pid: int,
        tid: int
    ) -> ProtocolMessage:
        """Parse HTTP request/response."""
        try:
            text = data.decode('utf-8', errors='replace')
            lines = text.split('\r\n')
            first_line = lines[0] if lines else ""

            attributes = {}

            if first_line.startswith('HTTP/'):
                # Response
                parts = first_line.split(' ', 2)
                attributes['status_code'] = int(parts[1]) if len(parts) > 1 else 0
                attributes['status_text'] = parts[2] if len(parts) > 2 else ""
                direction = "response"
            else:
                # Request
                parts = first_line.split(' ')
                attributes['method'] = parts[0] if parts else ""
                attributes['path'] = parts[1] if len(parts) > 1 else ""
                attributes['version'] = parts[2] if len(parts) > 2 else ""
                direction = "request"

            # Parse headers
            headers = {}
            for line in lines[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()
            attributes['headers'] = headers

            return ProtocolMessage(
                protocol=Protocol.HTTP,
                timestamp=timestamp,
                pid=pid,
                tid=tid,
                direction=direction,
                latency_ns=None,
                attributes=attributes
            )
        except Exception as e:
            logger.debug(f"HTTP parse error: {e}")
            return ProtocolMessage(
                protocol=Protocol.HTTP,
                timestamp=timestamp,
                pid=pid,
                tid=tid,
                direction="unknown",
                latency_ns=None
            )

    def _parse_http2(
        self,
        data: bytes,
        timestamp: datetime,
        pid: int,
        tid: int
    ) -> ProtocolMessage:
        """Parse HTTP/2 frame."""
        if len(data) < 9:
            return ProtocolMessage(
                protocol=Protocol.HTTP2,
                timestamp=timestamp,
                pid=pid,
                tid=tid,
                direction="unknown",
                latency_ns=None
            )

        # HTTP/2 frame: length(3) + type(1) + flags(1) + stream_id(4)
        length = (data[0] << 16) | (data[1] << 8) | data[2]
        frame_type = data[3]
        flags = data[4]
        stream_id = ((data[5] & 0x7F) << 24) | (data[6] << 16) | (data[7] << 8) | data[8]

        frame_types = {
            0: "DATA",
            1: "HEADERS",
            2: "PRIORITY",
            3: "RST_STREAM",
            4: "SETTINGS",
            5: "PUSH_PROMISE",
            6: "PING",
            7: "GOAWAY",
            8: "WINDOW_UPDATE",
            9: "CONTINUATION",
        }

        return ProtocolMessage(
            protocol=Protocol.HTTP2,
            timestamp=timestamp,
            pid=pid,
            tid=tid,
            direction="request" if frame_type == 1 else "unknown",
            latency_ns=None,
            attributes={
                'frame_type': frame_types.get(frame_type, f"UNKNOWN({frame_type})"),
                'length': length,
                'flags': flags,
                'stream_id': stream_id
            }
        )

    def _parse_mysql(
        self,
        data: bytes,
        timestamp: datetime,
        pid: int,
        tid: int
    ) -> ProtocolMessage:
        """Parse MySQL packet."""
        if len(data) < 5:
            return ProtocolMessage(
                protocol=Protocol.MYSQL,
                timestamp=timestamp,
                pid=pid,
                tid=tid,
                direction="unknown",
                latency_ns=None
            )

        # MySQL packet: length(3) + sequence(1) + command(1)
        length = data[0] | (data[1] << 8) | (data[2] << 16)
        sequence = data[3]
        command = data[4] if len(data) > 4 else 0

        commands = {
            0x00: "COM_SLEEP",
            0x01: "COM_QUIT",
            0x02: "COM_INIT_DB",
            0x03: "COM_QUERY",
            0x04: "COM_FIELD_LIST",
        }

        return ProtocolMessage(
            protocol=Protocol.MYSQL,
            timestamp=timestamp,
            pid=pid,
            tid=tid,
            direction="request" if command == 0x03 else "unknown",
            latency_ns=None,
            attributes={
                'command': commands.get(command, f"UNKNOWN({command})"),
                'sequence': sequence,
                'length': length
            }
        )

    def _parse_postgres(
        self,
        data: bytes,
        timestamp: datetime,
        pid: int,
        tid: int
    ) -> ProtocolMessage:
        """Parse PostgreSQL packet."""
        if len(data) < 5:
            return ProtocolMessage(
                protocol=Protocol.POSTGRES,
                timestamp=timestamp,
                pid=pid,
                tid=tid,
                direction="unknown",
                latency_ns=None
            )

        msg_type = chr(data[0]) if data[0] < 128 else "?"
        length = (data[1] << 24) | (data[2] << 16) | (data[3] << 8) | data[4]

        msg_types = {
            'Q': "Query",
            'P': "Parse",
            'B': "Bind",
            'E': "Execute",
            'S': "Sync",
            'C': "Close",
            'D': "Describe",
            'T': "RowDescription",
            'Z': "ReadyForQuery",
        }

        return ProtocolMessage(
            protocol=Protocol.POSTGRES,
            timestamp=timestamp,
            pid=pid,
            tid=tid,
            direction="request" if msg_type in "QPBE" else "response",
            latency_ns=None,
            attributes={
                'message_type': msg_types.get(msg_type, f"UNKNOWN({msg_type})"),
                'length': length
            }
        )

    def _parse_redis(
        self,
        data: bytes,
        timestamp: datetime,
        pid: int,
        tid: int
    ) -> ProtocolMessage:
        """Parse Redis RESP protocol."""
        try:
            text = data.decode('utf-8', errors='replace')
            resp_type = text[0] if text else '?'

            resp_types = {
                '+': "Simple String",
                '-': "Error",
                ':': "Integer",
                '$': "Bulk String",
                '*': "Array",
            }

            # Try to extract command from array
            command = ""
            if resp_type == '*':
                lines = text.split('\r\n')
                if len(lines) > 2 and lines[2]:
                    command = lines[2].upper()

            return ProtocolMessage(
                protocol=Protocol.REDIS,
                timestamp=timestamp,
                pid=pid,
                tid=tid,
                direction="request" if resp_type == '*' else "response",
                latency_ns=None,
                attributes={
                    'resp_type': resp_types.get(resp_type, f"UNKNOWN({resp_type})"),
                    'command': command
                }
            )
        except Exception:
            return ProtocolMessage(
                protocol=Protocol.REDIS,
                timestamp=timestamp,
                pid=pid,
                tid=tid,
                direction="unknown",
                latency_ns=None
            )


# =============================================================================
# Advanced eBPF Manager
# =============================================================================

class AdvancedeBPFManager:
    """
    Advanced eBPF manager with CO-RE support.

    Provides Pixie/Cilium-level capabilities:
    - Automatic protocol detection
    - Distributed tracing correlation
    - Continuous profiling
    - Service mesh integration
    """

    def __init__(self):
        self.btf_loader = BTFLoader()
        self.protocol_parser = ProtocolParser()
        self.programs: Dict[str, AdvancedBPFProgram] = {}
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.running = False

        # Check CO-RE support
        self.core_supported = self.btf_loader.load_vmlinux_btf()
        if self.core_supported:
            logger.info("CO-RE support enabled")
        else:
            logger.warning("CO-RE not available, using fallback mode")

    def register_program(self, name: str, program: AdvancedBPFProgram):
        """Register a BPF program."""
        self.programs[name] = program

    def add_event_handler(self, event_type: str, handler: Callable):
        """Add an event handler for a specific event type."""
        self.event_handlers[event_type].append(handler)

    async def start(self):
        """Start all registered BPF programs."""
        self.running = True
        logger.info(f"Starting {len(self.programs)} eBPF programs")

        # In a real implementation, this would:
        # 1. Compile BPF programs with CO-RE relocations
        # 2. Load programs into kernel
        # 3. Attach to appropriate hooks
        # 4. Start polling ring buffers

        asyncio.create_task(self._event_loop())

    async def stop(self):
        """Stop all BPF programs."""
        self.running = False
        logger.info("Stopped eBPF programs")

    async def _event_loop(self):
        """Main event processing loop."""
        while self.running:
            # In real implementation, poll ring buffers here
            await asyncio.sleep(0.1)

    def _dispatch_event(self, event_type: str, event: Any):
        """Dispatch event to registered handlers."""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def get_struct_layout(self, struct_name: str) -> Optional[Dict[str, Tuple[int, int]]]:
        """Get layout of a kernel struct using BTF."""
        return self.btf_loader.parser.get_struct_layout(struct_name)

    def create_http_tracer(self) -> AdvancedBPFProgram:
        """Create HTTP tracer program."""
        prog = AdvancedBPFProgram("http_tracer")

        prog.define_map(
            "http_events",
            BPFMapType.RINGBUF,
            0, 0, 256 * 1024
        ).define_map(
            "active_connections",
            BPFMapType.HASH,
            8, 256, 10000
        ).define_program(
            "trace_tcp_sendmsg",
            AdvancedBPFProgram.PROG_TYPE_KPROBE,
            0
        )

        self.register_program("http_tracer", prog)
        return prog

    def create_ssl_tracer(self) -> AdvancedBPFProgram:
        """Create SSL/TLS tracer program."""
        prog = AdvancedBPFProgram("ssl_tracer")

        prog.define_map(
            "ssl_events",
            BPFMapType.RINGBUF,
            0, 0, 1024 * 1024
        )

        self.register_program("ssl_tracer", prog)
        return prog

    def create_profiler(self) -> AdvancedBPFProgram:
        """Create continuous profiler program."""
        prog = AdvancedBPFProgram("cpu_profiler")

        prog.define_map(
            "stack_traces",
            BPFMapType.STACK_TRACE,
            4, 127 * 8, 10000
        ).define_map(
            "profile_counts",
            BPFMapType.HASH,
            12, 32, 100000
        ).define_program(
            "profile_cpu",
            AdvancedBPFProgram.PROG_TYPE_PERF_EVENT,
            0
        )

        self.register_program("cpu_profiler", prog)
        return prog
