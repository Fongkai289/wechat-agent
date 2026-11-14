import json
import tempfile
import unittest
from pathlib import Path

from wechat_agent import qa_store

TURN = "a" * 32


class QaStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "qa_conversations.json"

    def tearDown(self):
        self.tmp.cleanup()

    def write_legacy(self):
        legacy = {"conversations": [
            {"id": "old", "title": "Old", "created_at": "2025-01-01T00:00:00", "updated_at": "2025-01-02T00:00:00",
             "messages": [{"role": "user", "content": "q1"},
                          {"role": "assistant", "content": "a1", "turn_id": TURN, "pending": True}]},
            "not-a-conversation",
            {"id": "tie-first", "title": "", "created_at": "", "updated_at": "2025-01-03T00:00:00", "messages": []},
            {"id": "tie-second", "title": "T2", "created_at": "", "updated_at": "2025-01-03T00:00:00"},
        ]}
        self.path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        return self.path.read_bytes()

    def test_legacy_json_is_imported_once_and_left_untouched(self):
        original = self.write_legacy()
        rows = qa_store.list_qa_conversations(self.path)
        self.assertEqual([row["id"] for row in rows], ["tie-first", "tie-second", "old"])
        self.assertEqual(rows[0]["title"], "新对话")
        self.assertEqual(rows[2]["message_count"], 2)
        self.assertEqual(qa_store.get_qa_conversation(self.path, "old")["messages"][0]["content"], "q1")
        self.assertEqual(self.path.read_bytes(), original)
        # Deleting everything must not trigger a second import of the backup.
        for row in rows:
            self.assertTrue(qa_store.delete_qa_conversation(self.path, row["id"]))
        self.assertEqual(qa_store.list_qa_conversations(self.path), [])

    def test_interrupt_marks_imported_pending_answers(self):
        self.write_legacy()
        qa_store.interrupt_pending_qa(self.path)
        message = qa_store.get_qa_conversation(self.path, "old")["messages"][-1]
        self.assertNotIn("pending", message)
        self.assertEqual(message["error"], "服务已重启")

    def test_update_turn_rewrites_only_that_message(self):
        conv = qa_store.save_qa_conversation(self.path, "c1", [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "working", "turn_id": TURN, "pending": True},
            {"role": "user", "content": "later"},
        ])
        self.assertEqual(conv["title"], "q")
        updated = qa_store.update_qa_turn(self.path, "c1", TURN,
                                          {"role": "assistant", "content": "done", "turn_id": TURN})
        self.assertEqual([m["content"] for m in updated["messages"]], ["q", "done", "later"])
        self.assertNotIn("pending", updated["messages"][1])
        with self.assertRaises(ValueError):
            qa_store.update_qa_turn(self.path, "missing", TURN, {})

    def test_save_keeps_latest_messages_and_custom_title(self):
        qa_store.save_qa_conversation(self.path, "c1", [{"role": "user", "content": "first"}])
        qa_store.rename_qa_conversation(self.path, "c1", "Mine")
        messages = [{"role": "user", "content": str(i)} for i in range(250)]
        saved = qa_store.save_qa_conversation(self.path, "c1", messages)
        self.assertEqual(saved["title"], "Mine")
        stored = qa_store.get_qa_conversation(self.path, "c1")["messages"]
        self.assertEqual(len(stored), qa_store.MAX_STORED_MESSAGES)
        self.assertEqual(stored[0]["content"], "50")

    def test_database_files_are_owner_only(self):
        qa_store.create_qa_conversation(self.path, "x")
        db = qa_store.database_path(self.path)
        for candidate in (db, Path(str(db) + "-wal"), Path(str(db) + "-shm")):
            if candidate.exists():
                self.assertEqual(candidate.stat().st_mode & 0o777, 0o600, candidate.name)

    def test_store_is_recreated_if_database_file_is_removed(self):
        qa_store.create_qa_conversation(self.path, "x")
        qa_store.database_path(self.path).unlink()
        for suffix in ("-wal", "-shm"):
            Path(str(qa_store.database_path(self.path)) + suffix).unlink(missing_ok=True)
        self.assertEqual(qa_store.list_qa_conversations(self.path), [])


if __name__ == "__main__":
    unittest.main()
