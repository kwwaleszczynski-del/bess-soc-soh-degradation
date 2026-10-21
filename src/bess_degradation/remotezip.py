"""Extract single members of a large remote zip file with HTTP range requests, without downloading the whole archive.

A zip file ends with a central directory that lists every member with its size and the offset of its local header.
Reading the last 64 kB gives the end-of-central-directory record, one more range request the directory itself, and
then each wanted member costs one request for its local header plus one for its (deflated) bytes.

Usage:
    from bess_degradation.remotezip import RemoteZip
    rz = RemoteZip("https://zenodo.org/records/22939281/files/Data.zip?download=1")
    [m.name for m in rz.members]
    rz.extract("Data/Capacities in checkups - all cells.xlsx", Path("data/cells"))
"""
from __future__ import annotations

import logging
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

import requests

log = logging.getLogger(__name__)
USER_AGENT = "bess-soc-soh-degradation (open-source research; github)"


@dataclass(frozen=True)
class Member:
    name: str
    method: int            # 0 stored, 8 deflate
    compressed_size: int
    size: int
    header_offset: int


def parse_central_directory(cd: bytes) -> list[Member]:
    """Members listed in a raw central directory (handles zip64 extra fields for sizes and offsets)."""
    out, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[p + 10:p + 12])[0]
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        n, m, k = struct.unpack("<HHH", cd[p + 28:p + 34])
        offset = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + n].decode("utf-8", "replace")
        extra = cd[p + 46 + n:p + 46 + n + m]
        if 0xFFFFFFFF in (csize, usize, offset):
            csize, usize, offset = _zip64(extra, usize, csize, offset)
        out.append(Member(name, method, csize, usize, offset))
        p += 46 + n + m + k
    return out


def _zip64(extra: bytes, usize: int, csize: int, offset: int) -> tuple[int, int, int]:
    q = 0
    while q + 4 <= len(extra):
        tag, size = struct.unpack("<HH", extra[q:q + 4])
        if tag == 0x0001:
            vals, r = [], q + 4
            for v in (usize, csize, offset):
                if v == 0xFFFFFFFF:
                    vals.append(struct.unpack("<Q", extra[r:r + 8])[0])
                    r += 8
                else:
                    vals.append(v)
            return vals[1], vals[0], vals[2]
        q += 4 + size
    return csize, usize, offset


class RemoteZip:
    """A zip file behind a URL that supports HTTP range requests (``fetch(start, end)`` can be injected for tests)."""

    def __init__(self, url: str, fetch: Callable[[int, int], bytes] | None = None, size: int | None = None):
        self.url = url
        self._session = None
        if fetch is None:
            self._session = requests.Session()
            self._session.headers["User-Agent"] = USER_AGENT
            fetch = self._http_range
        self._fetch = fetch
        self.size = size if size is not None else self._remote_size()
        self.members = self._read_directory()

    def _remote_size(self) -> int:
        r = self._session.head(self.url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        return int(r.headers["Content-Length"])

    def _http_range(self, start: int, end: int, retries: int = 6) -> bytes:
        for attempt in range(retries):
            try:
                r = self._session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
                r.raise_for_status()
                if r.status_code != 206:
                    raise RuntimeError("server ignored the Range header")
                if len(r.content) != end - start + 1:
                    raise requests.RequestException(f"short read: {len(r.content)} of {end - start + 1} bytes")
                return r.content
            except requests.RequestException as e:      # Zenodo answers bursts of requests with HTTP 429/500
                if attempt == retries - 1:
                    raise
                wait = 10 * 2 ** attempt
                log.warning("range request failed (%s), retrying in %d s", e, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def _read_directory(self) -> list[Member]:
        tail_len = min(self.size, 65536 + 22)
        tail = self._fetch(self.size - tail_len, self.size - 1)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise ValueError("end of central directory not found")
        cd_size, cd_offset = struct.unpack("<II", tail[i + 12:i + 20])
        if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
            j = tail.rfind(b"PK\x06\x06")
            cd_size, cd_offset = struct.unpack("<QQ", tail[j + 40:j + 56])
        return parse_central_directory(self._fetch(cd_offset, cd_offset + cd_size - 1))

    def read(self, name: str) -> bytes:
        """Bytes of one member, decompressed."""
        m = next((x for x in self.members if x.name == name), None)
        if m is None:
            raise KeyError(name)
        head = self._fetch(m.header_offset, m.header_offset + 29)
        n, k = struct.unpack("<HH", head[26:30])
        start = m.header_offset + 30 + n + k
        raw = self._fetch(start, start + m.compressed_size - 1) if m.compressed_size else b""
        if m.method == 0:
            data = raw
        elif m.method == 8:
            data = zlib.decompress(raw, -15)
        else:
            raise NotImplementedError(f"compression method {m.method}")
        if len(data) != m.size:
            raise ValueError(f"{name}: expected {m.size} bytes, got {len(data)}")
        return data

    def extract(self, name: str, dest_dir: Path, overwrite: bool = False) -> Path:
        """Write one member to ``dest_dir`` (keeping its folder structure); skipped if it already exists."""
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe member name: {name}")
        out = dest_dir.joinpath(*rel.parts)
        if out.exists() and not overwrite:
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        log.info("extracting %s", name)
        tmp = out.with_name(out.name + ".part")
        tmp.write_bytes(self.read(name))
        tmp.replace(out)
        return out
