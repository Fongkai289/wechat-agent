import json
import sqlite3
import unittest
from unittest.mock import Mock, patch

from tests import test_qa_agent as fixtures
from tests.test_qa_agent import NOW, decision, completion
from wechat_agent import web
from wechat_agent.model_ids import ModelIDs, model_id
from wechat_agent.platform_tools import action_from_arguments
from wechat_agent.qa_agent import run_conversation_agent
from wechat_agent.qa_review import review_format, parse_review
from wechat_agent.qa_task import resolve_task


class ExclusionTests(unittest.TestCase):
    def setUp(self):
        fixtures.ToolTests.setUp(self)
        self.conn.row_factory = sqlite3.Row
    insert = fixtures.ToolTests.insert

    def test_private_peer_and_group_sender_excluded_before_keyword_limit_and_context(self):
        for i in range(1,26): self.insert(i,person='alice')
        self.insert(26,person='bob',chat='other')
        self.insert(27,person='alice',chat='group@chatroom')
        self.insert(28,person='bob',chat='group@chatroom')
        self.conn.execute("UPDATE messages SET chat_type='private' WHERE id<=26")
        self.conn.execute("UPDATE messages SET sender_username='me' WHERE id=1")
        self.conn.commit()
        scope={'exclude_person_ids':['alice']}
        hits=web.query_qa_search_like(self.conn,'internship',[],None,1,scope=scope)
        self.assertEqual(len(hits),1)
        self.assertNotEqual(hits[0]['sender_username'],'alice')
        plan={**self.plan,**scope,'people':[]}
        rows=self.tools.timeline(plan)['items']
        self.assertEqual({r['local_id'] for r in rows},{'26','28'})
        self.hybrid.return_value=(rows,{'rerank_unit':'conversation_window'})
        result=self.tools.search(plan,size=1)
        self.assertEqual({r['local_id'] for r in result['items']},{'26','28'})

    def test_semantic_candidates_exclude_private_peer_before_top_k(self):
        self.insert(1,person='alice');self.insert(2,person='bob')
        self.conn.execute('CREATE TABLE semantic_chunks(id INTEGER PRIMARY KEY)')
        self.conn.executemany('INSERT INTO semantic_chunks VALUES(?)',[(1,),(2,)])
        self.conn.executemany('INSERT INTO semantic_message_map VALUES(?,?)',[(1,1),(2,2)])
        self.conn.commit()
        rows=web.select_semantic_candidate_rows(self.conn,[],None,scope={'exclude_person_ids':['alice']})
        self.assertEqual([r['id'] for r in rows],[2])

    def test_nested_exclusion_ids_are_typed_and_validated(self):
        ids=ModelIDs([],[{'people':[{'id':'alice','name':'Alice'}]}])
        filters={'chat_type':'all','chat_keyword':'','since':None,'until':None,
                 'exclude_person_ids':[model_id('person','alice')]}
        args=ids.decode_arguments({'chat_ids':[],'person_ids':[],'query':'topic','size':10,'filters':filters})
        action=action_from_arguments('search',args,[],[{'people':[{'id':'alice'}]}],self.plan)
        self.assertEqual(action['filters']['exclude_person_ids'],['alice'])
        with self.assertRaises(ValueError):action_from_arguments('search',args,[],[],self.plan)


class QualityTests(unittest.TestCase):
    def test_optional_reference_does_not_block_supported_limited_answer(self):
        task={'request':'比较候选','expected_result':'说明依据','requirements':[
            {'id':'optional','importance':'reference','text':'可参考电话'}]}
        checks={key:{'status':'satisfied','reason':'ok'} for key in
                review_format(task)['json_schema']['schema']['properties']['task_checks']['properties']}
        checks['optional']={'status':'needs_evidence','reason':'无电话记录'}
        payload=completion({'checks':[{'paragraph_index':1,'status':'supported','reason':'有依据的暂定候选','source_refs':[1]}],
            'coverage':{'status':'covered','reason':'已说明局限'},'task_checks':checks})
        self.assertEqual(parse_review(payload,1,1,task)['decision'],'accept')

    def test_legacy_task_reinterprets_original_user_intent(self):
        old={'version':1,'request':'必须有电话','requirements':[],
             'user_turns':['也可以参考电话','主要看聊天内容']}
        model=Mock(return_value=completion({'request':'主要按聊天内容比较','expected_result':'给候选',
            'clarification':'','add_requirements':[{'text':'电话为参考','importance':'reference',
                'source_turn':1,'source_quote':'也可以参考电话'}],'remove_requirements':[]}))
        task=resolve_task(model,'结论呢',[{'role':'assistant','retrieval':{'current_task':old}}],NOW)
        material=json.loads(model.call_args.args[0][-1]['content'])
        self.assertEqual(material['previous_task'],{})
        self.assertEqual(task['version'],2)
        self.assertEqual(task['requirements'][0]['importance'],'reference')

    def test_exclusion_persists_and_zero_gain_does_not_force_finish(self):
        tools=Mock();tools.state.since_ts=None
        tools.find_people.return_value={'items':[],'people':[{'id':'alice','name':'Alice'}],'complete':True}
        tools.search.return_value={'items':[{'evidence_id':'one','text':'bob message'}],'complete':False}
        args={'chat_type':'all','chat_keyword':'','since':None,'until':None,'exclude_person_ids':['alice']}
        first=decision('search',query='first',filters=args)
        first['choices'][0]['message']['tool_calls'][0]['function']['arguments']=json.dumps({
            'query':'first','chat_ids':[],'person_ids':[],'size':20,
            'filters':{**args,'exclude_person_ids':[model_id('person','alice')]}})
        choose=Mock(side_effect=[decision('find_people',query='Alice'),first,
                                decision('search',query='second'),decision('search',query='third'),decision('finish')])
        with patch('wechat_agent.qa_agent.review_answer',return_value={'decision':'accept','issues':[]}):
            _,_,diag=run_conversation_agent('排除Alice后比较',[],[],tools,choose,
                Mock(return_value={'paragraphs':[]}),Mock(),Mock(),lambda:None,now=NOW)
        self.assertEqual(choose.call_count,5)
        self.assertEqual(tools.search.call_args.args[0]['exclude_person_ids'],['alice'])
        self.assertEqual([t['new_evidence_count'] for t in diag['trace'] if t['tool']=='search'],[1,0,0])
