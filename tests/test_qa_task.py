import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests.test_qa_agent import NOW, completion, decision
from wechat_agent.qa_task import resolve_task, apply_task_update, task_inputs
from wechat_agent.qa_agent import run_conversation_agent
from wechat_agent.qa_review import parse_review, review_format
from wechat_agent import web
from types import SimpleNamespace


def update(request='比较其他联系人', **changes):
    return dict(request=request,expected_result='比较并给出聊天证据',clarification='',
                add_requirements=[],remove_requirements=[],**changes)


class TaskTests(unittest.TestCase):
    def initial(self):
        turns=['8月以来谁和我亲近','排除甲，再选几个']
        value=update()
        value['add_requirements']=[{'text':'排除甲','source_turn':2,'source_quote':'排除甲'}]
        return apply_task_update(value,None,turns)

    def test_unmentioned_requirement_survives_followup_and_storage(self):
        old=self.initial()
        turns=old['user_turns']+['主要看聊天内容']
        value=update('8月以来，排除甲，主要按聊天内容比较其他联系人')
        value['add_requirements']=[{'text':'主要看聊天内容','source_turn':3,'source_quote':turns[-1]}]
        task=apply_task_update(value,old,turns)
        self.assertEqual([r['text'] for r in task['requirements']],['排除甲','主要看聊天内容'])
        msg=web.sanitize_qa_messages_for_store([{'role':'assistant','content':'answer',
            'retrieval':{'current_task':task,'model_metrics':[{'phase':'task'}]}}])[0]
        previous,ledger=task_inputs('所以结论呢',[msg])
        self.assertEqual(previous,task)
        self.assertEqual(ledger,turns+['所以结论呢'])
        self.assertIn('model_metrics',msg['retrieval'])

    def test_removal_requires_new_user_evidence_and_valid_requirement(self):
        old=self.initial(); value=update()
        row={'id':old['requirements'][0]['id'],'source_turn':2,'source_quote':'排除甲'}
        value['remove_requirements']=[row]
        with self.assertRaisesRegex(ValueError,'新的用户'): apply_task_update(value,old,old['user_turns']+['结论呢'])
        row.update(source_turn=3,source_quote='这次包括甲')
        new=apply_task_update(value,old,old['user_turns']+['这次包括甲'])
        self.assertEqual(new['requirements'],[])

    def test_fabricated_quote_cannot_create_a_requirement(self):
        value=update(); value['add_requirements']=[{'text':'只查私聊','source_turn':1,'source_quote':'只查私聊'}]
        with self.assertRaisesRegex(ValueError,'原话'): apply_task_update(value,None,['查群聊'])

    def test_legacy_history_is_read_in_full_even_beyond_ten_messages(self):
        history=[{'role':'user','content':'排除甲'}]+[{'role':'user','content':f'追问{i}'} for i in range(15)]
        model=Mock(return_value=completion(update()))
        resolve_task(model,'结论呢',history,NOW)
        body=json.loads(model.call_args.args[0][-1]['content'])
        self.assertEqual(body['user_turns'][0]['text'],'排除甲')
        self.assertEqual(len(body['user_turns']),17)

    def test_first_turn_requires_no_extra_model_call(self):
        model=Mock()
        task=resolve_task(model,'查询实习',[],NOW)
        self.assertEqual(task['request'],'查询实习'); model.assert_not_called()

    def test_clarification_does_not_search_or_answer_from_unrelated_evidence(self):
        data=update(); data['clarification']='这次继续比较联系人，还是换一个问题？'
        model=Mock(return_value=completion(data)); answer=Mock()
        tools=SimpleNamespace(state=SimpleNamespace(since_ts=None))
        result,sources,diag=run_conversation_agent('这个呢',[{'role':'user','content':'比较联系人'}],[],
            tools,model,answer,Mock(),Mock(),lambda:None,now=NOW)
        self.assertEqual(result['paragraphs'][0]['text'],data['clarification'])
        self.assertEqual(sources,[]); self.assertEqual(diag['quality_status'],'needs_clarification')
        answer.assert_not_called()

    def test_shared_task_reaches_search_answer_and_review_and_resumes_without_replanning(self):
        task=self.initial()
        history=[{'role':'assistant','content':'previous','retrieval':{'current_task':task}}]
        tools=SimpleNamespace(state=SimpleNamespace(since_ts=None),timeline=Mock(return_value={
            'items':[{'evidence_id':'one','text':'evidence'}],'complete':True}))
        models=Mock(side_effect=[completion(update()),decision('timeline'),decision('finish')])
        answer=Mock(side_effect=TimeoutError('unknown'))
        with tempfile.TemporaryDirectory() as d, patch('wechat_agent.qa_agent.review_answer',return_value={'decision':'accept','issues':[]}) as review:
            path=Path(d)/'turn.db'
            def run(**kw):
                return run_conversation_agent('结论呢',history,[],tools,models,answer,Mock(),Mock(),lambda:None,
                                              now=NOW,checkpoint_path=path,**kw)
            with self.assertRaises(TimeoutError): run()
            self.assertIn('排除甲',answer.call_args.args[0])
            body=json.loads(models.call_args_list[1].args[0][-1]['content'])
            self.assertEqual(body['current_task']['requirements'],task['requirements'])
            before=models.call_count
            answer.side_effect=None; answer.return_value={'paragraphs':[]}
            _,_,diag=run(resume=True,confirm_retry=True)
            self.assertEqual(models.call_count,before)
            self.assertEqual(review.call_args.kwargs['current_task'],diag['current_task'])

    def test_requirement_violation_overrides_otherwise_positive_review(self):
        task=self.initial(); contract=review_format(task)
        checks={key:{'status':'satisfied','reason':'ok'} for key in contract['json_schema']['schema']['properties']['task_checks']['properties']}
        checks[task['requirements'][0]['id']]={'status':'needs_answer','reason':'答案包含了明确排除的对象'}
        payload=completion({'checks':[{'paragraph_index':1,'status':'supported','reason':'quote exists','source_refs':[1]}],
                            'coverage':{'status':'covered','reason':'answered'},'task_checks':checks})
        self.assertEqual(parse_review(payload,1,1,task)['decision'],'revise')
        del checks['expected_result']
        with self.assertRaises(ValueError):
            parse_review(completion({'checks':[],'coverage':{'status':'covered','reason':'ok'},'task_checks':checks}),1,0,task)
