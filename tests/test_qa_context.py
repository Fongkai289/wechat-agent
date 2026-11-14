import copy
import json
import unittest

from wechat_agent.qa_context import decision_evidence, compact_tool_history


class CompactContextTests(unittest.TestCase):
    def test_catalog_and_window_links_are_lossless_without_repeated_long_ids(self):
        rows = [{'evidence_id':f'message-source-with-a-long-identity-{i:04d}',
                 'chat_id':'group', 'sender':'Alice', 'timestamp':i,
                 'text':f'message {i}', 'context_ids':[]} for i in range(300)]
        for i, row in enumerate(rows):
            row['context_ids'] = [x['evidence_id'] for x in rows[max(0,i-12):i+13] if x is not row]
        original = copy.deepcopy(rows)
        compact = decision_evidence(rows)
        refs = [ref for block in compact['chronological_blocks'] for ref in block['ordered_refs']]
        self.assertEqual(refs,list(range(1,301)))
        self.assertNotIn('context_windows',compact)  # Full graph stays in host.
        self.assertEqual(len(compact['source_catalog']),len(rows))
        self.assertEqual(len(compact['evidence']),len(rows))
        self.assertLess(len(json.dumps(compact)),len(json.dumps(rows))*.5)
        self.assertEqual(rows,original)

    def test_old_tool_history_is_compacted_without_breaking_pairing_or_mutation(self):
        history = [{'role':'assistant','tool_calls':[{'id':'call-1','function':{'name':'search'}}]},
                   {'role':'tool','tool_call_id':'call-1','content':json.dumps({
                       'items':[{'text':'unique-original','context_ids':['very-long-id']*100}],
                       'tool':'search','complete':False,'next_offset':1})}]
        original = copy.deepcopy(history)
        compact = compact_tool_history(history)
        self.assertEqual(compact[0],history[0])
        self.assertEqual(compact[1]['tool_call_id'],'call-1')
        payload = json.loads(compact[1]['content'])
        self.assertEqual(payload['returned'],1)
        self.assertEqual(payload['next_offset'],1)
        self.assertNotIn('unique-original',compact[1]['content'])
        self.assertEqual(history,original)

    def test_preview_budget_is_explicit_and_does_not_remove_catalog_entries(self):
        data = decision_evidence([{'evidence_id':str(i),'text':'abcdef'} for i in range(10)],text_budget=7)
        self.assertEqual(len(data['source_catalog']),10)
        self.assertEqual(sum(len(r['text']) for r in data['evidence']),7)
        self.assertTrue(data['evidence'][-1]['preview_truncated'])

    def test_historical_tool_text_cannot_bypass_preview_budget(self):
        rows = [{'evidence_id':'old','text':'older evidence'}, {'evidence_id':'new','text':'newest'}]
        history = [{'role':'tool','content':json.dumps({'items':[
            {'evidence_id':'old','text':'older evidence','preview_truncated':False},
            {'evidence_id':'gone','text':'not in working set'}]})}]*2
        view = decision_evidence(rows,text_budget=6,tool_messages=history)
        self.assertEqual({p['reference']:p['text'] for p in view['evidence']}, {1:'old',2:'new'})
        self.assertEqual(sum(len(p['text']) for p in view['evidence']),6)
        self.assertTrue(all(p['preview_truncated'] for p in view['evidence']))
        self.assertEqual(len(view['evidence']),2)
