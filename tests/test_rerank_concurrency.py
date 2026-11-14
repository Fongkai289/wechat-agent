import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from wechat_agent import web, jobs
from wechat_agent.goal_checkpoints import checkpoint_session, GoalResumeRequired
from wechat_agent.qa_recovery import CallReceipts, durable_model_call
from wechat_agent.retrieval_execution import map_rerank_batches


def scores(messages):
    rows = json.loads(messages[1]['content'])['candidates']
    return json.dumps({'scores':{row['id']:{'score':0.5,'reason':'fixture'} for row in rows}})


class RerankConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.state = SimpleNamespace(llm_config=Path('/unused'))
        self.config = {'qa':{}, 'rerank':{'enabled':True,'batch_size':8}}
        for name, value in [('load_llm_config',self.config),('effective_llm_profile',{}),
                            ('resolve_llm_api_key','synthetic')]:
            p = patch.object(web, name, return_value=value)
            p.start()
            self.addCleanup(p.stop)

    def run_search(self, complete, count=24):
        rows = [{'local_id':str(i), 'text':str(i)} for i in range(count)]
        return web.rerank_qa_items_with_config(self.state, 'test', rows, count, complete=complete)

    def test_three_concurrent_batches_complete_with_stable_ties(self):
        barrier = threading.Barrier(3, timeout=3)
        lock = threading.Lock()
        active = maximum = 0
        def complete(profile, key, messages, **options):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            barrier.wait()
            time.sleep(.01)
            with lock:
                active -= 1
            return scores(messages)
        rows, diagnostics = self.run_search(complete, 48)
        self.assertEqual(maximum, 3)
        self.assertEqual([r['local_id'] for r in rows], [str(i) for i in range(48)])
        self.assertEqual(diagnostics['rerank_batches'],6)
        self.assertEqual([b['batch'] for b in diagnostics['rerank_batch_timings']],list(range(1,7)))

    def test_partial_failure_reuses_completed_batches_and_requires_confirmation(self):
        calls, fail = {}, True
        lock = threading.Lock()
        @durable_model_call
        def fake_model(profile, api_key, messages, timeout=180, response_format=None):
            key = json.loads(messages[1]['content'])['candidates'][0]['text']
            with lock:
                calls[key] = calls.get(key,0)+1
            if key == '8' and fail:
                time.sleep(.03)
                raise TimeoutError('unknown response')
            return scores(messages)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'receipts.db'
            def run(confirm=False):
                with checkpoint_session(path) as (_, store):
                    receipts = CallReceipts(store.conn, confirm)
                    with receipts.scope('execute:1/search'):
                        return self.run_search(fake_model)
            _, diag = run()
            self.assertFalse(diag['rerank_complete'])
            self.assertEqual(calls, {'0':1,'8':1,'16':1})
            fail = False
            with self.assertRaises(GoalResumeRequired):
                run()
            self.assertEqual(calls, {'0':1,'8':1,'16':1})
            _, diag = run(True)
            self.assertTrue(diag['rerank_complete'])
            self.assertEqual(calls, {'0':1,'8':2,'16':1})

    def test_repair_resume_does_not_repeat_first_paid_response(self):
        calls, fail = [], True
        @durable_model_call
        def fake_model(profile, api_key, messages, timeout=180, response_format=None):
            calls.append(len(messages))
            if len(messages)==2:
                return '{"scores":{}}'
            if fail:
                raise TimeoutError('repair result unknown')
            return scores(messages)
        with checkpoint_session() as (_, store):
            def run(confirm=False):
                receipts = CallReceipts(store.conn, confirm)
                with receipts.scope('search'):
                    return self.run_search(fake_model, 8)
            run()
            fail = False
            with self.assertRaises(GoalResumeRequired): run()
            _, diag = run(True)
            self.assertEqual(calls,[2,3,3])
            self.assertTrue(diag['rerank_complete'])

    def test_legacy_serial_receipt_names_are_kept_on_resume(self):
        calls = []
        @durable_model_call
        def call_chat_payload(messages):
            calls.append(messages)
            return 'done'
        with checkpoint_session() as (_, store):
            receipts = CallReceipts(store.conn)
            with receipts.scope('search'):
                call_chat_payload('batch-0')
            with receipts.scope('search'):
                result = list(map_rerank_batches([('a','batch-0'),('b','batch-1')],
                    lambda batch, check: call_chat_payload(batch)))
            self.assertEqual(calls,['batch-0','batch-1'])
            self.assertEqual(result,[(0,'done'),(1,'done')])

    def test_skill_search_recovery_reuses_nested_paid_batch_receipts(self):
        from tests.test_code_skills import package
        from wechat_agent.code_skill_service import run_code_skill, synthetic_api
        calls, fail = {}, True
        lock = threading.Lock()
        @durable_model_call
        def fake_model(profile, api_key, messages, timeout=180, response_format=None):
            key = json.loads(messages[1]['content'])['candidates'][0]['text']
            with lock: calls[key] = calls.get(key,0)+1
            if key == '8' and fail:
                time.sleep(.03)
                raise TimeoutError('unknown batch')
            return scores(messages)
        def workflow(package, context, dispatch, **kwargs):
            dispatch('tool', {'name':'search','arguments':{
                'query':'test','chat_ids':[],'person_ids':[],'size':24}})
            return {'summary':'Done','items':[],'notice':None}
        def backend(*args):
            _, diagnostics = self.run_search(fake_model)
            if not diagnostics['rerank_complete']:
                raise TimeoutError(diagnostics['rerank_error'])
            return {'items':[],'complete':False}
        with tempfile.TemporaryDirectory() as directory:
            goal = {'skill':package(),'run_id':'test','profile_signature':'test',
                    'checkpoint_path':str(Path(directory)/'skill.db')}
            def run():
                api = synthetic_api(goal['skill'])
                api.tool_backend = backend
                try:
                    return run_code_skill(goal,api,lambda data:None,lambda:False,'test')
                finally: api.close()
            with patch('wechat_agent.code_skill_service.run_package', side_effect=workflow):
                with self.assertRaises(TimeoutError): run()
                self.assertEqual(calls,{'0':1,'8':1,'16':1})
                goal.update(resume=True)
                with self.assertRaises(GoalResumeRequired): run()
                self.assertEqual(calls,{'0':1,'8':1,'16':1})
                fail = False
                goal['confirm_retry'] = True
                self.assertEqual(json.loads(run())['summary'],'Done')
                self.assertEqual(calls,{'0':1,'8':2,'16':1})

    def test_worker_receives_parent_cancellation_and_does_not_start_later_batches(self):
        job = jobs.Job('test','test',{},'test')
        jobs._current.job = job
        self.addCleanup(setattr, jobs._current, 'job', None)
        gate = threading.Barrier(3, timeout=3)
        started = []
        def invoke(batch, check):
            started.append(batch)
            gate.wait()
            job.cancel.set()
            check()
        with self.assertRaises(jobs.JobCancelled):
            list(map_rerank_batches([(str(i),i) for i in range(12)],invoke))
        self.assertEqual(sorted(started),[0,1,2])
