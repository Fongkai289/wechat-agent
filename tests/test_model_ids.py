"""Model identity namespaces are shared by Q&A and native Skill calls."""
import copy
import json
import unittest
from unittest.mock import Mock

from tests.test_code_skills import package
from tests.test_qa_agent import decision
from wechat_agent.code_skill_service import synthetic_api
from wechat_agent.model_ids import ModelIDs, model_id
from wechat_agent.platform_tools import FUNCTION_TOOLS, parse_tool_call


class ModelIdentityTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{'chat_id':'alice', 'chat_title':'Alice', 'sender_username':'me',
                         'sender':'Me', 'evidence_id':'message:1', 'text':'original'}]
        self.observations = [{'chats':[{'chat_id':'alice', 'chat_title':'Alice', 'senders':[
            {'sender_username':'alice'}, {'sender_username':'me'}]}],
            'people':[{'id':'alice', 'name':'Alice'}]}]
        self.ids = ModelIDs(self.sources, self.observations)

    def test_same_raw_identity_has_distinct_chat_and_person_handles(self):
        self.assertNotEqual(model_id('chat','alice'), model_id('person','alice'))
        args = {'chat_ids':[model_id('chat','alice')], 'person_ids':[model_id('person','me')],
                'source_ids':[model_id('evidence','message:1')]}
        self.assertEqual(self.ids.decode_arguments(args),
                         {'chat_ids':['alice'], 'person_ids':['me'], 'source_ids':['message:1']})

    def test_sender_cannot_be_used_as_chat_even_when_raw_value_is_shared(self):
        for invalid in ('alice', 'me', model_id('person','alice'), model_id('person','me'),
                        model_id('chat','me'), model_id('chat','hidden')):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'chat_'):
                self.ids.decode_arguments({'chat_ids':[invalid]})
        with self.assertRaisesRegex(ValueError, 'person_'):
            self.ids.decode_arguments({'person_ids':[model_id('chat','alice')]})
        with self.assertRaisesRegex(ValueError, 'evidence_'):
            self.ids.decode_arguments({'source_ids':[model_id('chat','alice')]})

    def test_scalar_chat_field_uses_same_namespace(self):
        self.assertEqual(self.ids.decode_arguments({'chat_id':model_id('chat','alice')}), {'chat_id':'alice'})
        with self.assertRaises(ValueError):
            self.ids.decode_arguments({'chat_id':model_id('person','alice')})

    def test_structured_catalog_and_nested_tool_results_are_typed(self):
        original = copy.deepcopy(self.observations)
        data = {'observations': self.observations, 'source_catalog':self.sources,
                'windows':[{'source_ids':['message:1']}], 'text':'alice me message:1'}
        encoded = self.ids.encode(data)
        chat = encoded['observations'][0]['chats'][0]
        self.assertEqual(chat['chat_id'], model_id('chat','alice'))
        self.assertEqual(chat['senders'][0]['sender_username'], model_id('person','alice'))
        self.assertEqual(encoded['observations'][0]['people'][0]['id'], model_id('person','alice'))
        self.assertEqual(encoded['source_catalog'][0]['evidence_id'], model_id('evidence','message:1'))
        self.assertEqual(encoded['windows'][0]['source_ids'], [model_id('evidence','message:1')])
        self.assertEqual(encoded['text'], data['text'])
        self.assertEqual(self.observations, original)

    def test_wire_history_preserves_pairing_and_database_values(self):
        messages = [{'role':'user', 'content':'alice is a name in this prose'},
                    {'role':'assistant','content':None,'tool_calls':[{'id':'call_1','type':'function',
                     'function':{'name':'timeline','arguments':json.dumps({'chat_ids':['alice']})}}]},
                    {'role':'tool','tool_call_id':'call_1','content':json.dumps({'items':self.sources})}]
        original = copy.deepcopy(messages)
        wire = self.ids.messages(messages)
        assistant = next(m for m in wire if m.get('tool_calls'))
        result = next(m for m in wire if m['role']=='tool')
        self.assertEqual(assistant['tool_calls'][0]['id'], result['tool_call_id'])
        self.assertEqual(json.loads(assistant['tool_calls'][0]['function']['arguments'])['chat_ids'], [model_id('chat','alice')])
        self.assertEqual(json.loads(result['content'])['items'][0]['sender_username'], model_id('person','me'))
        self.assertEqual(messages, original)
        self.assertIn(messages[0], wire)

    def test_request_schemas_separate_allowed_sets_and_do_not_mutate_catalog(self):
        original = copy.deepcopy(FUNCTION_TOOLS)
        definitions = self.ids.definitions(FUNCTION_TOOLS)
        search = next(t['function']['parameters']['properties'] for t in definitions if t['function']['name']=='search')
        self.assertEqual(search['chat_ids']['items']['pattern'], '^chat_[0-9a-f]{24}$')
        self.assertEqual(search['person_ids']['items']['pattern'], '^person_[0-9a-f]{24}$')
        self.assertEqual(FUNCTION_TOOLS, original)
        empty = ModelIDs([], []).definitions(FUNCTION_TOOLS)
        search = next(t['function']['parameters']['properties'] for t in empty if t['function']['name']=='search')
        self.assertEqual(search['chat_ids']['maxItems'], 0)

    def test_large_catalog_keeps_type_constraints_and_checks_exact_membership(self):
        ids = ModelIDs([], [{'chats':[{'chat_id':str(i)} for i in range(1200)]}])
        definitions = ids.definitions(FUNCTION_TOOLS)
        search = next(t['function']['parameters']['properties'] for t in definitions if t['function']['name']=='search')
        self.assertEqual(search['chat_ids']['items']['pattern'], '^chat_[0-9a-f]{24}$')
        self.assertEqual(ids.decode_arguments({'chat_ids':[model_id('chat','1199')]}), {'chat_ids':['1199']})
        with self.assertRaises(ValueError):
            ids.decode_arguments({'chat_ids':[model_id('chat','1200')]})

    def test_restore_rebuilds_stable_handles_without_changing_membership(self):
        restored = ModelIDs(json.loads(json.dumps(self.sources)), json.loads(json.dumps(self.observations)))
        self.assertEqual(restored.known, self.ids.known)
        removed = ModelIDs([], self.observations)
        with self.assertRaises(ValueError):
            removed.decode_arguments({'source_ids':[model_id('evidence','message:1')]})

    def test_decoded_call_is_validated_by_existing_scope_checks(self):
        response = decision('search',chat_ids=['alice'],person_ids=['me'],query='call')
        raw = self.ids.decode_response(response)
        action, assistant = parse_tool_call(raw,self.sources,self.observations,{'until_ts':1000},[])
        self.assertEqual(action['chat_ids'], ['alice'])
        self.assertEqual(action['person_ids'], ['me'])
        self.assertIn('chat_', response['choices'][0]['message']['tool_calls'][0]['function']['arguments'])

    def test_sql_translates_bound_handles_not_query_text(self):
        sql = 'SELECT * FROM messages WHERE chat_id=? AND sender_username=?'
        args = {'sql':sql, 'params':[model_id('chat','alice'),model_id('person','me')]}
        self.assertEqual(self.ids.decode_arguments(args), {'sql':sql, 'params':['alice','me']})


class NativeSkillIdentityTests(unittest.TestCase):
    def setUp(self):
        self.api = synthetic_api(package())
        self.addCleanup(self.api.close)
        self.api.api_chats()
        self.api.api_messages('alice')

    def test_model_uses_handles_while_existing_skill_code_gets_raw_arguments(self):
        self.api.backends['llm'] = Mock(return_value=(decision('timeline',chat_ids=['alice'],person_ids=['alice'])['choices'][0],{}))
        messages = [{'role':'user','content':json.dumps({'items':self.api.available_sources()})}]
        response = self.api.api_llm(messages,tools=['timeline'])
        args = json.loads(response['message']['tool_calls'][0]['function']['arguments'])
        self.assertEqual(args['chat_ids'], ['alice'])
        result = self.api.api_tool('timeline', args)
        self.assertEqual(result['items'][0]['chat_id'], 'alice')
        wire = self.api.backends['llm'].call_args.args[0]
        self.assertEqual(json.loads(wire[-1]['content'])['items'][0]['chat_id'], model_id('chat','alice'))

    def test_model_raw_or_cross_type_parameters_are_rejected_before_execution(self):
        for invalid in ('alice', model_id('person','alice')):
            payload = decision('timeline')['choices'][0]
            fn = payload['message']['tool_calls'][0]['function']
            args = json.loads(fn['arguments']); args['chat_ids'] = [invalid]; fn['arguments'] = json.dumps(args)
            self.api.backends['llm'] = Mock(return_value=(payload,{}))
            with self.assertRaisesRegex(ValueError, 'chat_'):
                self.api.api_llm([{'role':'user','content':'Read Alice'}],tools=['timeline'])

    def test_unreturned_chat_page_does_not_become_a_valid_model_handle(self):
        api = synthetic_api(package())
        self.addCleanup(api.close)
        api.api_tool('read_chats', {'offset':0,'size':1})
        ids = ModelIDs(api.available_sources(), api.tool_observations)
        with self.assertRaises(ValueError):
            ids.decode_arguments({'chat_ids':[model_id('chat','bob')]})
