"""Provider contracts and local validation agree across Q&A and Skills."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from wechat_agent import web
from wechat_agent.model_output import TEXT_FORMAT, output_contract, parse_structured_completion
from wechat_agent.platform_runtime import PlatformModel
from wechat_agent.platform_sql import SQLScopeError
from wechat_agent.code_skill_service import synthetic_api
from tests.test_code_skills import package


def response(value, finish='stop'):
    return {'choices':[{'finish_reason':finish,'message':{'content':json.dumps(value)}}],
            'usage':{'total_tokens':7}}


class StructuredTransportTests(unittest.TestCase):
    def test_duplicate_tool_arguments_rejected_before_id_decoding(self):
        from wechat_agent.model_ids import ModelIDs
        from wechat_agent.platform_tools import parse_tool_call
        payload={'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[
            {'id':'call1','type':'function','function':{'name':'read_chats',
             'arguments':'{"offset":0,"offset":1,"size":20}'}}]}}]}
        with self.assertRaisesRegex(ValueError,'重复'):
            ModelIDs([],[]).decode_response(payload)
        with self.assertRaisesRegex(ValueError,'重复'):
            parse_tool_call(payload,[],[],{},[])

    def test_missing_last_score_repair_names_the_field_and_preserves_contract(self):
        candidates=[{'local_id':str(i),'text':'fixture'} for i in range(48)]
        calls=[]
        def complete(profile,key,messages,**options):
            calls.append(len(messages))
            self.assertIn('c48',options['response_format']['json_schema']['schema']['properties']['scores']['required'])
            scores={f'c{i}':{'score':.5,'reason':''} for i in range(1,49)}
            if len(calls)==1: del scores['c48']
            else: self.assertIn('c48',messages[-1]['content'])
            return json.dumps({'scores':scores})
        with patch.object(web,'load_llm_config',return_value={'qa':{},'rerank':{'enabled':True}}), \
             patch.object(web,'effective_llm_profile',return_value={}), \
             patch.object(web,'resolve_llm_api_key',return_value='synthetic'):
            rows,diag=web.rerank_qa_items_with_config(SimpleNamespace(llm_config=None),'query',candidates,48,complete)
        self.assertEqual(len(rows),48)
        self.assertTrue(diag['rerank_complete'])
        self.assertEqual(calls,[2,3])

    def test_agent_passes_local_error_back_to_model_without_claiming_no_matches(self):
        from tests.test_qa_agent import decision, NOW
        from wechat_agent.qa_agent import run_conversation_agent
        calls=[]
        def complete(messages,schema,**options):
            calls.append(messages)
            if len(calls)==1:
                return decision('query_messages',sql='SELECT count(*) FROM messages',params=[],since=None,until=None)
            self.assertIn('sql_scope_too_large',json.dumps(messages))
            return decision('finish')
        result={'tool':'query_messages','items':[],'complete':False,'matched':None,
                'error':{'code':'sql_scope_too_large','retryable':True},'note':'query was not executed'}
        with patch('wechat_agent.qa_agent.execute_action',return_value=result):
            _,sources,diag=run_conversation_agent('question',[],[],SimpleNamespace(state=SimpleNamespace(since_ts=None)),
                complete,lambda *a:None,lambda *a,**kw:None,lambda *a:None,lambda:None,now=NOW)
        self.assertEqual(len(calls),2)
        self.assertEqual(sources,[])
        self.assertFalse(diag['coverage']['exhaustive'])

    def test_local_scope_error_becomes_incomplete_tool_receipt(self):
        api=synthetic_api(package())
        self.addCleanup(api.close)
        with patch.object(api,'execute_data',side_effect=SQLScopeError('scope too large')):
            result=api.api_tool('query_messages',dict(sql='SELECT count(*) FROM messages',params=[],since=None,until=None,offset=0,size=20))
        self.assertFalse(result['complete'])
        self.assertIsNone(result['matched'])
        self.assertEqual(result['error']['code'],'sql_scope_too_large')

    def test_paid_timeout_is_not_converted_to_empty_search(self):
        api=synthetic_api(package())
        self.addCleanup(api.close)
        with patch.object(api,'execute_data',side_effect=TimeoutError('unknown billed result')):
            with self.assertRaises(TimeoutError):
                api.api_tool('query_messages',dict(sql='SELECT count(*) FROM messages',params=[],since=None,until=None,offset=0,size=20))

    def test_each_candidate_is_required_including_last(self):
        contract = web.rerank_output_format(48)
        scores = {f'c{i}':{'score':.5,'reason':'relevant'} for i in range(1,49)}
        self.assertEqual(parse_structured_completion(response({'scores':scores}),contract)['scores'],scores)
        del scores['c48']
        with self.assertRaisesRegex(ValueError,'c48'):
            parse_structured_completion(response({'scores':scores}),contract)

    def test_unknown_fields_wrong_types_and_duplicates_fail(self):
        for item in ({'score':'0.5','reason':''}, {'score':True,'reason':''},
                     {'score':1.1,'reason':''}, {'scope':.5,'reason':''},
                     {'score':.5,'reason':'','extra':1}):
            with self.subTest(item=item), self.assertRaises(ValueError):
                parse_structured_completion(response({'scores':{'c1':item}}),web.rerank_output_format(1))
        payload=response({})
        payload['choices'][0]['message']['content']='{"text":"a","text":"b"}'
        with self.assertRaisesRegex(ValueError,'重复'):
            parse_structured_completion(payload,TEXT_FORMAT)

    def test_refusal_truncation_and_nonfinite_values_fail(self):
        for finish in ('length',None,'tool_calls'):
            with self.assertRaises(ValueError):
                parse_structured_completion(response({'text':'partial'},finish),TEXT_FORMAT)
        payload=response({'text':'denied'})
        payload['choices'][0]['message']['refusal']='declined'
        with self.assertRaises(ValueError): parse_structured_completion(payload,TEXT_FORMAT)
        for invalid in ('NaN','Infinity','1e999'):
            payload=response({})
            payload['choices'][0]['message']['content']='{"scores":{"c1":{"score":'+invalid+',"reason":""}}}'
            with self.assertRaises(ValueError):
                parse_structured_completion(payload,web.rerank_output_format(1))

    def test_skill_plain_text_keeps_public_return_type(self):
        model=PlatformModel({}, {}, lambda:None)
        with patch.object(model,'complete',return_value=response({'text':'hello'})) as complete:
            value, usage=model.llm([{'role':'user','content':'question'}])
        self.assertEqual(value,'hello')
        self.assertEqual(usage['total_tokens'],7)
        self.assertEqual(complete.call_args.args[1],TEXT_FORMAT)

    def test_skill_schema_validated_before_program_receives_result(self):
        model=PlatformModel({}, {}, lambda:None)
        with patch.object(model,'complete',return_value=response({'text':42})):
            with self.assertRaises(ValueError): model.llm([],schema=TEXT_FORMAT['json_schema']['schema'])

    def test_no_unstructured_downgrade_or_shared_schema_mutation(self):
        for contract in ({'type':'json_object'}, {'type':'json_schema','json_schema':None}):
            with self.assertRaises(ValueError): output_contract(contract)
        with self.assertRaises(ValueError): output_contract(tools=[{'type':'function','function':None}])
        clone=output_contract()
        clone['json_schema']['schema']['properties']['text']['type']='number'
        self.assertEqual(TEXT_FORMAT['json_schema']['schema']['properties']['text']['type'],'string')
