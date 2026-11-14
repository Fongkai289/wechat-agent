"""Topic recall must not turn into title routing; query branches merge once."""
import copy
import json
import sqlite3
import threading
import unittest
from unittest.mock import Mock,patch

from tests import test_qa_agent as fixture
from wechat_agent import web
from wechat_agent.platform_tools import FUNCTION_TOOLS,action_from_arguments
from wechat_agent.qa_agent import run_conversation_agent
from wechat_agent.retrieval_execution import parallel_recall
from wechat_agent.qa_recovery import CallReceipts,recorded_call


class MultiQueryTests(unittest.TestCase):
    setUp = fixture.ToolTests.setUp
    insert = fixture.ToolTests.insert

    def setup_messages(self):
        self.insert(1,chat='alice');self.insert(2,chat='bob');self.insert(3,chat='other@chatroom')
        self.conn.executemany('UPDATE messages SET text=?,search_text=? WHERE id=?',[
            ('portfolio reference','portfolio reference',1),('reference','reference',2),('portfolio','portfolio',3)])
        self.conn.execute(web.FTS_SCHEMA)
        self.conn.execute('INSERT INTO messages_fts(rowid,search_text) SELECT id,search_text FROM messages')
        self.conn.commit()
        self.state.llm_config=self.state.qa_search_db.parent/'models.json'

    def test_schema_distinguishes_objective_queries_and_literal_title(self):
        spec=next(t['function']['parameters']['properties'] for t in FUNCTION_TOOLS if t['function']['name']=='search')
        props=spec['filters']['anyOf'][1]['properties']
        self.assertNotIn('chat_keyword',props)
        self.assertIn('整段文字',props['chat_title_contains']['description'])
        self.assertIn('queries',spec)
        args={'query':'application help','queries':['portfolio','reference','portfolio'],
              'chat_ids':[],'person_ids':[],'size':20,
              'filters':{'chat_type':'all','chat_title_contains':'','since':None,'until':None}}
        result=action_from_arguments('search',args,[],[],self.plan)
        self.assertEqual(result['queries'],['portfolio','reference'])
        args['queries']=['']
        with self.assertRaises(ValueError):action_from_arguments('search',args,[],[],self.plan)

    def test_saved_skill_legacy_field_keeps_title_semantics(self):
        args={'query':'portfolio','chat_ids':[],'person_ids':[],'size':20,
              'filters':{'chat_type':'all','chat_keyword':'alice','since':None,'until':None}}
        action=action_from_arguments('search',args,[],[],self.plan)
        self.assertEqual(action['filters']['chat_title_contains'],'alice')
        self.assertEqual(action['queries'],[])

    def test_empty_title_scope_is_explained_without_paid_recall(self):
        self.setup_messages()
        plan={**self.plan,'people':[],'chat_scope':{'type':'all','keyword':'portfolio reference'}}
        result=self.tools.search(plan,'portfolio')
        self.hybrid.assert_not_called()
        self.assertEqual(result['error']['code'],'empty_chat_scope')
        self.assertEqual(result['scope_diagnostics']['matching_chats'],0)
        self.assertFalse(result['scope_diagnostics']['body_search_executed'])
        self.assertFalse(result['complete'])

    def test_union_deduplicates_and_reranks_once_with_original_objective(self):
        self.setup_messages()
        plan={'queries':[{'text':'portfolio'},{'text':'reference'}],
              'verified_people':True,'chat_scope':{'type':'private','keyword':''}}
        def rerank(state,query,windows,count):
            self.assertEqual(query,'Who helped my application?')
            members=[m for w in windows for m in w['retrieval_members']]
            self.assertEqual(sorted(m['local_id'] for m in members),['1','2'])
            return windows,{'rerank_count':len(windows)}
        with patch.object(web,'qa_search_db_status',return_value={'usable':True}), \
             patch.object(web,'query_qa_semantic_index',return_value=([],{})), \
             patch.object(web,'rerank_qa_items_with_config',side_effect=rerank) as rank:
            rows,diag=web.search_qa_search_db(self.state,'Who helped my application?',[],None,20,plan)
        rank.assert_called_once()
        self.assertEqual({r['chat_id'] for r in rows},{'alice','bob'})
        self.assertEqual([q['query'] for q in diag['query_results']],['portfolio','reference'])
        self.assertTrue(all(q['fts_count'] for q in diag['query_results']))

    def test_semantic_queries_each_get_quota_and_embedding_is_one_batch(self):
        self.setup_messages()
        self.conn.execute('CREATE TABLE semantic_chunks(id INTEGER PRIMARY KEY,embedding BLOB)')
        self.conn.executemany('INSERT INTO semantic_chunks VALUES (?,?)',[
            (1,web.pack_float_vector([1,0])),(2,web.pack_float_vector([0,1])),(3,web.pack_float_vector([.9,.1]))])
        self.conn.executemany('INSERT INTO semantic_message_map VALUES (?,?)',[(1,1),(2,2),(3,3)])
        self.conn.commit()
        config=copy.deepcopy(web.DEFAULT_LLM_CONFIG)
        with patch.object(web,'load_llm_config',return_value=config), \
             patch.object(web,'qa_embedding_index_status',return_value={'ready':True}), \
             patch.object(web,'resolve_llm_api_key',return_value='synthetic'), \
             patch.object(web,'call_embeddings',return_value=([[1,0],[0,1]],{'total_tokens':5})) as embed, \
             patch.object(web,'qa_item_from_semantic_row',side_effect=lambda r,*a:{'semantic_chunk_id':r['id']}):
            rows,diag=web.query_qa_semantic_index(self.state,'help',[],None,2,
                {'queries':['portfolio','reference'],'chat_scope':{'type':'private','keyword':''}})
        embed.assert_called_once()
        self.assertEqual(embed.call_args.args[2],['portfolio','reference'])
        self.assertEqual({r['semantic_chunk_id'] for r in rows},{1,2})
        self.assertEqual([q['returned'] for q in diag['query_results']],[1,1])

    def test_two_empty_searches_can_correct_scope_on_third_attempt(self):
        self.setup_messages()
        self.hybrid.return_value=([web.qa_item_from_search_row(dict(zip(
            [d[0] for d in self.conn.execute('SELECT * FROM messages').description],
            self.conn.execute('SELECT * FROM messages WHERE id=1').fetchone())))],{})
        filters={'chat_type':'all','chat_title_contains':'portfolio reference','since':None,'until':None}
        choose=Mock(side_effect=[fixture.decision('search',query='portfolio',filters=filters),
            fixture.decision('search',query='reference',filters=filters),
            fixture.decision('search',query='help',queries=['portfolio','reference']),fixture.decision('finish')])
        answer=Mock(return_value={'paragraphs':[]})
        _,sources,diag=run_conversation_agent('application help',[],[],self.tools,choose,answer,Mock(),Mock(),lambda:None,now=fixture.NOW)
        self.assertTrue(sources)
        self.assertEqual(choose.call_count,4)
        self.hybrid.assert_called_once()
        request=json.loads(choose.call_args_list[2].args[0][-1]['content'])
        self.assertIn('empty_chat_scope',json.dumps(request))
        self.assertEqual(self.hybrid.call_args.kwargs['scope']['queries'],[{'text':'portfolio'},{'text':'reference'}])


class RecallExecutionTests(unittest.TestCase):
    def test_concurrent_branches_and_completed_paid_receipts_are_reused(self):
        conn=sqlite3.connect(':memory:',check_same_thread=False);self.addCleanup(conn.close)
        receipts=CallReceipts(conn)
        barrier=threading.Barrier(2)
        calls=[]
        def branch(name):
            def invoke():
                def paid():
                    barrier.wait(timeout=2);calls.append(name);return {'name':name}
                return recorded_call('embed',paid,True)
            return invoke
        branches=[('a',branch('a')),('b',branch('b'))]
        with receipts.scope('search'):first=parallel_recall(branches)
        with receipts.scope('search'):second=parallel_recall(branches)
        self.assertEqual(first,second)
        self.assertEqual(sorted(calls),['a','b'])
