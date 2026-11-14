"""In-memory, normalized embedding matrices for local semantic search.

Stored chunk vectors are decoded once per index version instead of once per
query, and cosine similarity is a single matrix-vector product. The cache is
invalidated by the SQLite file (and WAL sidecar) signature, so index updates
from another thread or process are picked up on the next query.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

FLOAT_BYTES = 4
MAX_CACHED_DATABASES = 2


def pack_float_vector(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack_float_vector(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).tolist()


def vector_norm(vector: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))


class VectorMatrix:
    """Unit-normalized vectors of one dimensionality, ordered by chunk id."""

    def __init__(self, ids: np.ndarray, matrix: np.ndarray):
        self.ids = ids
        self.matrix = matrix

    def positions(self, chunk_ids: Iterable[int] | None) -> np.ndarray:
        if chunk_ids is None:
            return np.arange(len(self.ids))
        wanted = np.fromiter((int(value) for value in chunk_ids), dtype=np.int64)
        if not len(wanted) or not len(self.ids):
            return np.empty(0, dtype=np.int64)
        index = np.searchsorted(self.ids, wanted)
        index = np.clip(index, 0, len(self.ids) - 1)
        return np.unique(index[self.ids[index] == wanted])


class SemanticVectorIndex:
    def __init__(self, signature: tuple, by_dimension: dict[int, VectorMatrix]):
        self.signature = signature
        self.by_dimension = by_dimension

    @property
    def size(self) -> int:
        return sum(len(block.ids) for block in self.by_dimension.values())

    def search(
        self,
        query: list[float],
        chunk_ids: Iterable[int] | None,
        limit: int | None,
    ) -> tuple[list[tuple[float, int]], int]:
        """Return ``(score, chunk_id)`` pairs with score >= 0, best first.

        Ties keep ascending chunk id order. Chunks whose stored dimensionality
        differs from the query are ignored, as before. The second value is the
        number of non-negative matches before ``limit`` was applied.
        """
        if limit is not None and limit <= 0:
            return [], 0
        vector = np.asarray(query, dtype=np.float32)
        block = self.by_dimension.get(len(vector))
        norm = float(np.linalg.norm(vector))
        if block is None or not norm:
            return [], 0
        positions = block.positions(chunk_ids)
        if not len(positions):
            return [], 0
        scores = block.matrix[positions] @ (vector / norm)
        keep = scores >= 0
        scores, ids = scores[keep], block.ids[positions[keep]]
        total = len(scores)
        if limit is not None and limit < total:
            # Partition first so only the best `limit` candidates (plus ties at
            # the boundary) are fully sorted.
            boundary = np.partition(scores, total - limit)[total - limit]
            keep = scores >= boundary
            scores, ids = scores[keep], ids[keep]
        order = np.lexsort((ids, -scores))
        if limit is not None:
            order = order[:limit]
        return [(float(scores[i]), int(ids[i])) for i in order], total


def database_signature(path: Path) -> tuple:
    signature: list[Any] = [str(path)]
    for candidate in (path, Path(str(path) + "-wal")):
        try:
            stat = os.stat(candidate)
        except OSError:
            signature.append(None)
            continue
        signature.append((stat.st_ino, stat.st_size, stat.st_mtime_ns))
    return tuple(signature)


def load_vector_index(conn: sqlite3.Connection, signature: tuple) -> SemanticVectorIndex:
    counts = conn.execute(
        "SELECT length(embedding), count(*) FROM semantic_chunks GROUP BY length(embedding)"
    ).fetchall()
    blocks: dict[int, tuple[np.ndarray, np.ndarray, list[int]]] = {}
    for byte_length, count in counts:
        if not byte_length or byte_length % FLOAT_BYTES:
            continue
        dimension = byte_length // FLOAT_BYTES
        blocks[dimension] = (
            np.empty(count, dtype=np.int64),
            np.empty((count, dimension), dtype=np.float32),
            [0],
        )
    # Stream rows so only one decoded copy of the index is ever resident.
    for chunk_id, blob in conn.execute("SELECT id, embedding FROM semantic_chunks ORDER BY id"):
        if not blob or len(blob) % FLOAT_BYTES:
            continue
        entry = blocks.get(len(blob) // FLOAT_BYTES)
        if entry is None:
            continue
        ids, matrix, cursor = entry
        row = cursor[0]
        if row >= len(ids):
            continue  # Rows inserted between the two statements.
        ids[row] = int(chunk_id)
        matrix[row] = np.frombuffer(blob, dtype=np.float32)
        cursor[0] = row + 1
    by_dimension = {}
    for dimension, (ids, matrix, cursor) in blocks.items():
        ids, matrix = ids[: cursor[0]], matrix[: cursor[0]]
        norms = np.linalg.norm(matrix, axis=1)
        # Zero vectors score 0, matching the previous scalar implementation.
        np.divide(matrix, norms[:, None], out=matrix, where=norms[:, None] > 0)
        matrix[norms == 0] = 0
        by_dimension[dimension] = VectorMatrix(ids, matrix)
    return SemanticVectorIndex(signature, by_dimension)


_cache: "OrderedDict[str, SemanticVectorIndex]" = OrderedDict()
_cache_lock = threading.Lock()
_build_locks: dict[str, threading.Lock] = {}


def cached_vector_index(path: Path, conn: sqlite3.Connection) -> SemanticVectorIndex:
    key = str(Path(path).resolve())
    with _cache_lock:
        build_lock = _build_locks.setdefault(key, threading.Lock())
    with build_lock:
        signature = database_signature(Path(key))
        with _cache_lock:
            current = _cache.get(key)
            if current is not None and current.signature == signature:
                _cache.move_to_end(key)
                return current
        index = load_vector_index(conn, signature)
        with _cache_lock:
            _cache[key] = index
            _cache.move_to_end(key)
            while len(_cache) > MAX_CACHED_DATABASES:
                _cache.popitem(last=False)
        return index


def clear_vector_cache() -> None:
    with _cache_lock:
        _cache.clear()
