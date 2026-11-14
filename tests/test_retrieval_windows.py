import unittest
from wechat_agent.retrieval_windows import message_key, rerank_windows, expand_ranked_windows


def row(i,chat='a',text='message',timestamp=None):
    return {'chat_id':chat,'local_id':str(i),'timestamp':i if timestamp is None else timestamp,
            'text':text,'sender':'Alice','time':str(i),'fusion_score':1}


class WindowTests(unittest.TestCase):
    def test_diversification_preserves_scores_and_full_windows_from_two_chats(self):
        rows=[row(i,'a') for i in range(1,41)]+[row(41,'b'),row(42,'c')]
        windows=rerank_windows(rows,[[message_key(r) for r in rows[:40]]])
        before=[w.get('fusion_score') for w in windows]
        result=expand_ranked_windows(windows,10,diversify_by_chat=True)
        self.assertEqual({r['chat_id'] for r in result},{'a','b'})
        self.assertEqual(len(result),41)
        self.assertEqual([w.get('fusion_score') for w in windows],before)

    def test_120_chunks_2400_messages_become_120_scoring_units(self):
        rows=[row(i,chat=str(i//20)) for i in range(2400)]
        groups=[[message_key(r) for r in rows[i:i+20]] for i in range(0,2400,20)]
        windows=rerank_windows(rows,groups)
        self.assertEqual(len(windows),120)
        restored=expand_ranked_windows(windows,2400)
        self.assertEqual({message_key(r) for r in restored},{message_key(r) for r in rows})

    def test_lexical_overlap_does_not_duplicate_scoring_and_window_tail_is_kept(self):
        rows=[row(1,text='deadline?'),row(2,text='Friday'),row(3,text='Correction: Monday')]
        windows=rerank_windows(rows,[[message_key(r) for r in rows]])
        result=expand_ranked_windows(windows,1)
        self.assertEqual(len(result),3)
        self.assertEqual(result[-1]['text'],'Correction: Monday')
        self.assertEqual(len({r['ranked_window_id'] for r in result}),1)

    def test_windows_never_mix_chats_or_hide_long_tails(self):
        rows=[row(1,'a','x'*3500),row(2,'a','y'*3500),row(3,'b','end')]
        windows=rerank_windows(rows,[[message_key(r) for r in rows]])
        self.assertEqual(len(windows),3)
        self.assertIn('y'*3500,windows[1]['text'])
        self.assertTrue(all(len({r['chat_id'] for r in w['retrieval_members']})==1 for w in windows))

    def test_lexical_windows_respect_time_gap_and_order(self):
        rows=[row(3,timestamp=4000),row(2,timestamp=2),row(1,timestamp=1)]
        windows=rerank_windows(rows,[])
        self.assertEqual([[r['local_id'] for r in w['retrieval_members']] for w in windows],[['1','2'],['3']])

    def test_empty_or_missing_scope_members_are_not_reintroduced(self):
        visible=row(2); hidden=row(1,'other')
        windows=rerank_windows([visible],[[message_key(hidden),message_key(visible)]])
        self.assertEqual(expand_ranked_windows(windows,10)[0]['local_id'],'2')
        self.assertEqual(rerank_windows([],[[message_key(hidden)]]),[])
