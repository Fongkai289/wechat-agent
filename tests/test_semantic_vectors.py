import math
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path

from wechat_agent import semantic_vectors


def scalar_rank(query, rows, chunk_ids, limit):
    """Reference implementation: the previous per-row Python cosine loop."""
    query_norm = math.sqrt(sum(v * v for v in query))
    scored = []
    for chunk_id, vector in rows:
        if chunk_ids is not None and chunk_id not in chunk_ids:
            continue
        if len(vector) != len(query):
            continue
        norm = math.sqrt(sum(v * v for v in vector))
        score = 0.0 if not norm else sum(a * b for a, b in zip(query, vector)) / (query_norm * norm)
        if score >= 0:
            scored.append((score, chunk_id))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return scored[:limit] if limit is not None else scored, len(scored)


class SemanticVectorTests(unittest.TestCase):
    def setUp(self):
        semantic_vectors.clear_vector_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "index.db"
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("CREATE TABLE semantic_chunks(id INTEGER PRIMARY KEY, embedding BLOB)")

    def tearDown(self):
        self.conn.close()
        semantic_vectors.clear_vector_cache()
        self.tmp.cleanup()

    def insert(self, rows):
        self.conn.executemany("INSERT INTO semantic_chunks VALUES (?,?)",
                              [(chunk_id, semantic_vectors.pack_float_vector(vector)) for chunk_id, vector in rows])
        self.conn.commit()

    def test_matches_scalar_ranking_with_filters_ties_and_mixed_dimensions(self):
        rng = random.Random(7)
        rows = [(i, [rng.uniform(-1, 1) for _ in range(16)]) for i in range(1, 400)]
        rows += [(400, [0.0] * 16), (401, [1.0] * 8), (402, rows[0][1]), (403, rows[0][1])]
        self.insert(rows)
        stored = [(i, semantic_vectors.unpack_float_vector(semantic_vectors.pack_float_vector(v))) for i, v in rows]
        index = semantic_vectors.cached_vector_index(self.path, self.conn)
        query = [rng.uniform(-1, 1) for _ in range(16)]
        subset = {i for i, _ in rows if i % 3 == 0} | {1, 402, 403}
        for chunk_ids in (None, subset):
            for limit in (None, 1, 5, 37):
                expected, expected_total = scalar_rank(query, stored, chunk_ids, limit)
                actual, total = index.search(query, chunk_ids, limit)
                self.assertEqual(total, expected_total)
                self.assertEqual([chunk_id for _, chunk_id in actual], [chunk_id for _, chunk_id in expected])
                for (score, _), (reference, _) in zip(actual, expected):
                    self.assertAlmostEqual(score, reference, places=5)
        # Identical vectors tie and keep ascending chunk id order.
        top, _ = index.search(rows[0][1], None, 3)
        self.assertEqual([chunk_id for _, chunk_id in top], [1, 402, 403])

    def test_index_reloads_after_database_changes(self):
        self.insert([(1, [1, 0])])
        first = semantic_vectors.cached_vector_index(self.path, self.conn)
        self.assertIs(semantic_vectors.cached_vector_index(self.path, self.conn), first)
        self.insert([(2, [0, 1])])
        second = semantic_vectors.cached_vector_index(self.path, self.conn)
        self.assertIsNot(second, first)
        self.assertEqual([chunk_id for _, chunk_id in second.search([0, 1], None, 1)[0]], [2])

    def test_unknown_or_empty_candidates_and_dimension_mismatch(self):
        self.insert([(1, [1, 0])])
        index = semantic_vectors.cached_vector_index(self.path, self.conn)
        self.assertEqual(index.search([1, 0], [], None), ([], 0))
        self.assertEqual(index.search([1, 0], [99], None), ([], 0))
        self.assertEqual(index.search([1, 0, 0], None, None), ([], 0))
        self.assertEqual(index.search([0, 0], None, None), ([], 0))


if __name__ == "__main__":
    unittest.main()
