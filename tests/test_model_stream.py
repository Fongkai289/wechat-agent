"""Streaming preserves structured results, paid recovery and visible timing."""
import io
import json
import sqlite3
import unittest
from unittest.mock import patch

from wechat_agent import web
from wechat_agent.model_stream import observe_model_transport
from wechat_agent.qa_recovery import CallReceipts
from wechat_agent.goal_checkpoints import GoalResumeRequired


def chunk(delta=None, finish=None):
    return {'choices':[{'index':0,'delta':delta or {},'finish_reason':finish}]}


def sse(parts, done=True):
    data=': keepalive\n\n'+''.join('data: '+json.dumps(p,ensure_ascii=False)+'\n\n' for p in parts)
    if done: data+='data: [DONE]\n\n'
    response=io.BytesIO(data.encode())
    response.headers={'Content-Type':'text/event-stream; charset=utf-8'}
    return response


class StreamingTests(unittest.TestCase):
    profile={'model':'fixture','base_url':'https://example.test/v1'}

    def invoke(self):
        return web.call_chat_payload(self.profile,'synthetic',[],stream=True)

    def test_text_chunks_usage_and_safe_progress(self):
        parts=[chunk({'role':'assistant'}),chunk({'content':'{"text":"你'}),
               chunk({'content':'好"}'},'stop'),{'choices':[],'usage':{'prompt_tokens':12,'completion_tokens':4,'total_tokens':16}}]
        events=[]
        with patch.object(web.urllib.request,'urlopen',return_value=sse(parts)) as send, observe_model_transport(events.append):
            payload=self.invoke()
        body=json.loads(send.call_args.args[0].data)
        self.assertTrue(body['stream'])
        self.assertTrue(body['stream_options']['include_usage'])
        self.assertEqual(json.loads(payload['choices'][0]['message']['content']),{'text':'你好'})
        self.assertEqual(payload['usage']['total_tokens'],16)
        metrics=payload['_transport']
        self.assertLessEqual(metrics['first_event_ms'],metrics['first_content_ms'])
        self.assertLessEqual(metrics['first_content_ms'],metrics['total_ms'])
        self.assertGreater(metrics['request_chars'],0)
        self.assertEqual([events[0]['status'],events[-1]['status']],['waiting','completed'])
        self.assertNotIn('你好',json.dumps(events,ensure_ascii=False))
        self.assertNotIn('synthetic',json.dumps(events))

    def test_tool_argument_fragments_assemble_before_execution(self):
        parts=[chunk({'tool_calls':[{'index':0,'id':'call_a','type':'function',
                   'function':{'name':'search','arguments':'{"query":'}}]}),
               chunk({'tool_calls':[{'index':0,'function':{'arguments':'"实习"}'}}]}),
               chunk(finish='tool_calls')]
        with patch.object(web.urllib.request,'urlopen',return_value=sse(parts)):
            result=self.invoke()['choices'][0]
        self.assertEqual(result['finish_reason'],'tool_calls')
        tool=result['message']['tool_calls'][0]
        self.assertEqual(tool['id'],'call_a')
        self.assertEqual(json.loads(tool['function']['arguments']),{'query':'实习'})

    def test_partial_stream_stays_unknown_paid_receipt_until_confirmed(self):
        db=sqlite3.connect(':memory:');self.addCleanup(db.close)
        receipts=CallReceipts(db)
        events=[]
        partial=sse([chunk({'content':'{"text":"partial'})],done=False)
        with patch.object(web.urllib.request,'urlopen',return_value=partial) as send, observe_model_transport(events.append):
            with receipts.scope('answer'),self.assertRaisesRegex(RuntimeError,'中断'):
                self.invoke()
            with receipts.scope('answer'),self.assertRaises(GoalResumeRequired):
                self.invoke()
        self.assertEqual(send.call_count,1)
        self.assertEqual(events[-1]['status'],'failed')
        self.assertEqual(db.execute('SELECT status FROM qa_calls').fetchone()[0],'pending')

    def test_completed_receipt_does_not_repeat_paid_stream_on_resume(self):
        db=sqlite3.connect(':memory:');self.addCleanup(db.close)
        receipts=CallReceipts(db)
        with patch.object(web.urllib.request,'urlopen',return_value=sse([chunk({'content':'{"text":"ok"}'},'stop')])) as send:
            with receipts.scope('answer'):first=self.invoke()
            with receipts.scope('answer'):second=self.invoke()
        self.assertEqual(send.call_count,1)
        self.assertEqual(first,second)

    def test_json_compatible_provider_does_not_trigger_second_request(self):
        payload={'choices':[{'message':{'content':'{"text":"ok"}'},'finish_reason':'stop'}]}
        response=io.BytesIO(json.dumps(payload).encode());response.headers={'Content-Type':'application/json'}
        with patch.object(web.urllib.request,'urlopen',return_value=response) as send:
            result=self.invoke()
        self.assertEqual(send.call_count,1)
        self.assertFalse(result['_transport']['stream'])
        self.assertIsNone(result['_transport']['first_content_ms'])

    def test_finish_without_done_or_done_without_finish_are_incomplete(self):
        for response in (sse([chunk({'content':'x'},'stop')],done=False),sse([chunk({'content':'x'})])):
            with patch.object(web.urllib.request,'urlopen',return_value=response), self.assertRaisesRegex(RuntimeError,'中断'):
                self.invoke()

    def test_cancellation_is_checked_between_chunks(self):
        with patch.object(web.urllib.request,'urlopen',return_value=sse([chunk({'content':'x'},'stop')])), \
             patch.object(web,'check_job_cancelled',side_effect=[None,RuntimeError('cancelled')]):
            with self.assertRaisesRegex(RuntimeError,'cancelled'):self.invoke()
