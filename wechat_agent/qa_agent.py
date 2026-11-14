"""A bounded tool-first LangGraph agent with server-owned evidence.

SQLite checkpoints and call receipts allow explicit recovery without replaying
completed stages. Restarting a process never automatically retries model calls.
"""
from __future__ import annotations

import json
import hashlib
import time
from copy import deepcopy
from datetime import datetime
from typing import TypedDict

from .goal_checkpoints import checkpoint_session
from .qa_recovery import AGENT_VERSION, CallReceipts, frozen_input, recorded_call
from .qa_review import review_answer
from .qa_context import decision_evidence, compact_tool_history
from .model_ids import ModelIDs
from .model_stream import observe_model_transport
from .qa_task import resolve_task, task_view, task_instruction
from .evidence_context import merge_evidence, select_evidence, context_windows
from .platform_tools import (INSTRUCTIONS, FUNCTION_TOOLS, execute_action,
                             parse_tool_call, tool_result_message, call_may_bill, ToolArgumentError)


# 图内共享计划、证据和覆盖率；原生调用消息与调用 ID 一并保存，供恢复后续接。
class AgentState(TypedDict, total=False):
    plan: dict
    sources: list
    trace: list
    coverage: dict
    answer_data: dict
    usage: dict
    action: dict
    rounds: int
    observations: list
    stop_reason: str
    tool_messages: list
    tool_call_id: str
    platform_state: dict
    evidence_pool: list
    reviews: list
    review_rounds: int
    review_route: str
    review_feedback: dict
    quality_status: str
    active_step: dict
    model_metrics: list
    answer_timings: list
    current_task: dict
    search_stalls: int
    transport_calls: list


def limitation(text):
    return {"paragraphs": [{"kind": "limitation", "text": text, "source_refs": []}]}


def run_conversation_agent(question, history, people, tools, complete, answer, emit, checkpoint,
                           check, limit=60, now=None, checkpoint_path=None, resume=False,
                           confirm_retry=False, identity=None, context_char_budget=120000,
                           final_message_limit=None):
    now = now or datetime.now().astimezone()
    with checkpoint_session(checkpoint_path) as (saver, receipts):
        # 入口冻结问题、历史、时钟和版本；恢复读取原快照，并核对模型与数据来源标识。
        inputs = frozen_input(receipts.conn, {
            'question': question, 'history': history, 'people': people, 'limit': limit,
            'now': now.isoformat(), 'identity': identity,
            'agent_version': AGENT_VERSION,
            'context_char_budget': context_char_budget,
            'final_message_limit': final_message_limit,
        }, resume)
        if inputs.get("agent_version") != AGENT_VERSION:
            raise ValueError("旧版检索执行记录已停用，请重新提问；已保存的历史回答不受影响")
        return _run_agent(inputs, tools, complete, answer, emit, checkpoint, check,
                          saver, CallReceipts(receipts.conn, confirm_retry), resume)


def _run_agent(inputs, tools, complete, answer, emit, checkpoint, check, saver, receipts, resume):
    from langgraph.graph import StateGraph, START, END
    from langsmith import tracing_context

    question, history = inputs['question'], inputs['history']
    now = datetime.fromisoformat(inputs['now'])
    limit = max(10, min(int(inputs['limit']), 200))
    # Search targets count hits, not the neighbours needed to interpret them.
    # A row cap is opt-in for controlled evaluations, not production answers.
    final_message_limit = inputs['final_message_limit']
    char_budget = max(10000, min(int(inputs['context_char_budget']), 1000000))
    retrieval_limit = 800
    started = time.monotonic()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0,
             "scope": "tool_decisions_and_reviews"}
    model_metrics = []
    transport_calls = []
    live_state = {}

    def model(messages, schema=None, tool_definitions=None):
        check()
        if time.monotonic() - started > 900:
            raise TimeoutError("本轮检索达到 900 秒预算，已保存执行过程，请重试")
        # 回执复用已完成请求；结果未知的计费请求由恢复层要求明确确认后重试。
        tick = time.monotonic()
        result = recorded_call('complete', lambda: complete(messages, schema,tools=tool_definitions,tool_choice='required')
                               if tool_definitions is not None else complete(messages,schema), True)
        model_metrics.append({'phase':'decision' if tool_definitions is not None else 'task' if (schema or {}).get('json_schema',{}).get('name')=='current_conversation_task' else 'review',
                              'elapsed_ms':round((time.monotonic()-tick)*1000),
                              'request_chars':len(json.dumps(messages,ensure_ascii=False)),
                              'prompt_tokens':int((result.get('usage') or {}).get('prompt_tokens') or 0)})
        usage["calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] += int((result.get("usage") or {}).get(key) or 0)
        check()
        return result

    def model_trace(s):
        # UI excerpts must not duplicate evidence or consume the model's context.
        return [{k: v for k, v in row.items() if k not in ('preview', 'preview_total','rerank_batch_timings','timings')}
                for row in s.get('trace', [])]

    def diagnostics(s):
        plan = s.get("plan") or {}
        since, until = plan.get("since_ts"), plan.get("until_ts")
        time_scope = ""
        if since is not None:
            time_scope = f"{datetime.fromtimestamp(since).isoformat(sep=' ', timespec='seconds')} 至 {datetime.fromtimestamp(until).isoformat(sep=' ', timespec='seconds')}"
        scope = '按各步工具条件检索'
        chat_scope=plan.get('chat_scope') or {}
        if chat_scope.get('type') in ('group','private'):
            scope += ' · '+('群聊' if chat_scope['type']=='group' else '私聊')
        if chat_scope.get('keyword'):
            scope += ' · 会话名包含 '+chat_scope['keyword']
        return {"mode": "原生工具调用检索", "scope": scope,
                "time_scope": time_scope, "query_plan": plan, "trace": s.get("trace", []),
                "coverage": s.get("coverage", {}), "usage": dict(s.get('usage') or usage), "context_count": len(s.get("sources", [])),
                'answer_reviews':s.get('reviews',[]),'quality_status':s.get('quality_status','pending'),
                'active_step':s.get('active_step'),
                'current_task':s.get('current_task'),
                'model_metrics':s.get('model_metrics',[]), 'answer_timings':s.get('answer_timings',[]),
                'transport_calls':list(transport_calls),
                "conversation_state": {"people": plan.get("people", []), "query": plan.get("query", ""),
                                       'chat_scope':plan.get('chat_scope',{}),
                                       "requested_at": plan.get("requested_at", ""), "time_expression": plan.get("time_expression", "")}}

    def save(s, stage, phase=None, parameters=None):
        nonlocal live_state
        # 对外发布诊断与证据快照；图节点的持久化另由 LangGraph 检查点负责。
        check()
        s['active_step'] = {'phase':phase or 'transition','message':stage,
                            'started_at_ms':int(time.time()*1000),
                            'parameters':parameters or {}}
        live_state = s
        checkpoint(diagnostics(s), s.get("sources", []))
        emit("progress", message=stage, retrieval=diagnostics(s))
        return s

    def transport_progress(event):
        # Metadata only: partial JSON/citations/tool arguments are never shown
        # or executed. Persist live progress so reconnecting tabs see it too.
        check()
        event['phase'] = (live_state.get('active_step') or {}).get('phase','model')
        position = next((i for i,x in enumerate(transport_calls) if x['request_id']==event['request_id']),None)
        if position is None: transport_calls.append(event)
        else: transport_calls[position] = event
        labels = {'waiting':'已发送请求，等待模型响应','receiving':'正在接收模型响应',
                  'completed':'模型响应接收完成','failed':'模型响应中断'}
        message = labels.get(event['status'],'模型请求进行中')
        if event.get('output_chars'): message += f" · 已接收 {event['output_chars']} 字符"
        checkpoint(diagnostics(live_state),live_state.get('sources',[]))
        emit('progress',message=message,retrieval=diagnostics(live_state))

    def run_tool(s, name, call, parameters=None):
        check()
        if len(s.get("trace", [])) >= 8 or time.monotonic() - started > 900:
            raise TimeoutError("检索预算已用完，未继续扩大查询")
        tick = time.monotonic()
        save(s, f"正在执行 {name}", phase='tool', parameters=parameters)
        snapshot = getattr(type(tools), 'snapshot', None)
        def invoke():
            result = call()
            return {**result, '_platform_state':tools.snapshot()} if snapshot else result
        result = dict(recorded_call(name, invoke, call_may_bill('tool', {'name':name})))
        # A completed tool receipt must also restore its evidence ledger and budgets.
        platform_state = result.pop('_platform_state', None)
        if platform_state is not None:
            tools.restore(platform_state)
            s['platform_state'] = platform_state
        check()
        # 按稳定证据 ID 排除已收录消息，再应用总预算；工具分页信息保留在轨迹中。
        seen = {item["evidence_id"] for item in s.get("sources", [])}
        merged = merge_evidence(s.get('sources', []), result['items'])
        for source in merged:
            linked = result.get('context_links', {}).get(source['evidence_id'])
            if linked is not None:
                source['context_ids'] = list(dict.fromkeys([*source.get('context_ids', []), *linked]))
        s['sources'], dropped = select_evidence(merged, limit=retrieval_limit)
        retained = {item['evidence_id'] for item in s['sources']}
        included = len(retained-seen)
        if name=='search':
            s['search_stalls'] = s.get('search_stalls',0)+1 if included==0 else 0
        event = {"tool": name, "parameters": parameters or {}, "returned": len(result.get('people',result.get('chats',result["items"]))),
                 'unit':'个人物' if 'people' in result else '个会话' if 'chats' in result else '条',
                 "included": included, "matched": result.get("matched"),
                 "complete": bool(result.get("complete")) and not dropped,
                 "next_offset": result.get("next_offset"), "note": result.get("note", ""),
                 "elapsed_ms": round((time.monotonic() - tick) * 1000)}
        if name=='search':
            event['new_evidence_count'] = included
            event['evidence_by_chat'] = [{'chat_title':title,'count':sum(x.get('chat_title')==title for x in result['items'])}
                for title in dict.fromkeys(x.get('chat_title') for x in result['items'])]
        event['scope_diagnostics'] = result.get('scope_diagnostics')
        event['tool_call_id'] = s['tool_call_id']
        # Local UI previews are bounded and explicitly incomplete, not a second
        # evidence pool. Never publish prompts, API configuration or hidden reasoning.
        preview_rows = result.get('people', result.get('chats', result['items']))
        event['preview'] = [{k: str(item[k])[:800] for k in (
            'evidence_id','name','title','chat_title','sender','time','text')
            if item.get(k) is not None} for item in preview_rows[:5]]
        event['preview_total'] = len(preview_rows)
        detail = result.get('diagnostics') or {}
        event['query_results'] = detail.get('query_results',[])
        event['timings'] = detail.get('timings', {})
        event['rerank_batch_timings'] = detail.get('rerank_batch_timings', [])
        event['retrieval_counts'] = {key: detail[key] for key in (
            'message_candidate_count', 'window_candidate_count', 'rerank_count') if key in detail}
        for field in ('anchor_count', 'context_count'):
            if field in result:
                event[field] = result[field]
        s["trace"] = [*s.get("trace", []), event]
        # 任一工具未完整返回或证据被预算截断，均不能标为穷尽覆盖。
        s["coverage"] = {"exhaustive": all(t["complete"] for t in s["trace"]),
                         "context_limit": final_message_limit, "context_char_budget": char_budget,
                         "included": len(s["sources"]),
                         "note": "仅覆盖本地索引；相关性检索和上下文预算不代表遍历全部消息"}
        save(s, f"{name}：返回 {event['returned']} 条，纳入 {event['included']} 条")
        # Tool previews must not advertise IDs removed from the working set;
        # otherwise a legitimate finish selection could fail ID validation.
        if dropped:
            result = {**result, 'items':[item for item in result['items'] if item['evidence_id'] in retained],
                      'complete':False, 'note':result.get('note','')+'；工作集预算省略了部分完整上下文窗口'}
        return result

    def plan_node(s):
        save(s,'结合历史更新本轮任务',phase='task')
        task = resolve_task(model,question,history,now)
        boundary = {'tool_first': True, 'people': [], 'query': task['request'],
                    'since_ts': tools.state.since_ts, 'until_ts': int(now.timestamp()),
                    'chat_scope': {'type': 'all', 'keyword': ''}, 'requested_at': now.isoformat()}
        s = dict(s, plan=boundary, sources=[], trace=[],current_task=task)
        if task.get('clarification'):
            return dict(s,answer_data=limitation(task['clarification']),quality_status='needs_clarification')
        return save(s, '开始按需检索')

    def decide_node(s):
        s=dict(s)
        # Bound exploration without pretending to have exhaustive coverage.
        if s.get('rounds',0)>=8 or len(s.get('sources',[]))>=retrieval_limit:
            reason='达到工具调用或上下文预算，未声称完整检索'
            s['coverage']=dict(s.get('coverage',{}),exhaustive=False,note=reason)
            s['evidence_pool'] = list(s.get('sources',[]))
            return dict(s,action={'tool':'finish'},stop_reason=reason)
        request = {
            'question':question,'current_task':task_view(s['current_task']),'observations':s.get('observations',[]),
            'answer_review_feedback':{k:v for k,v in (s.get('review_feedback') or {}).items() if k in ('decision','issues')},
            'trace':[{k:t[k] for k in ('tool','parameters','returned','included','complete','note','new_evidence_count','evidence_by_chat') if k in t}
                     for t in s.get('trace',[])],
            'calls_remaining':8-s.get('rounds',0),
            'evidence_slots_remaining':retrieval_limit-len(s.get('sources',[])),
            'search_hit_target':limit,
            'answer_evidence_limit':final_message_limit,
            'answer_context_char_budget':char_budget,
            'exclude_person_ids':s['plan'].get('exclude_person_ids',[]),
            **decision_evidence(s.get('sources',[]), tool_messages=s.get('tool_messages',[]))}
        request.update(current_time=now.isoformat(), platform_boundary={
            'since_ts':s['plan']['since_ts'],'until_ts':s['plan']['until_ts']},
            conversation_history=[{'role':m.get('role'),'content':str(m.get('content') or '')[:2000]}
                for m in history[-2:] if m.get('role') in ('user','assistant') and not any(m.get(k) for k in ('error','pending','stopped'))])
        messages=[{'role':'system','content':INSTRUCTIONS},
                  {'role':'user','content':json.dumps(request,ensure_ascii=False)}]
        save(s,f"模型决定第 {s.get('rounds',0)+1} 步检索",phase='model')
        messages[0]['content'] += ('本轮完整任务和有效要求见current_task，最新短句不能覆盖这些要求。每轮调用一个原生函数工具，结束时调用finish。'
            '若复核要求补查，只有一次修正机会：先补取对应证据，再作答；若无法补证，明确缩小结论，不重复同一草稿。')
        messages[0]['content'] += ('用户明确排除某人时，先按该人的姓名确认身份，再在filters.exclude_person_ids中传入对应ID；不要把排除姓名拼进正文查询。'
            '需要比较不同联系人或会话时，可在search的filters中设置diversify_by_chat=true，避免只得到一个对象的材料。'
            '根据trace中的查询和新增证据选择下一步；补查应对应review_feedback提出的具体对象或证据缺口，避免反复改写同一组泛词。'
            '比较性回答的finish应保留多个候选人的比较证据；信息不足时可给有依据的暂定候选并说明局限，不强行宣布唯一第一。')
        messages = [messages[0], {'role':'user','content':question}, *compact_tool_history(s.get('tool_messages',[])), messages[1]]
        identities = ModelIDs(s.get('sources', []), s.get('observations', []))
        response = model(identities.messages(messages, include_directory=False), tool_definitions=identities.definitions(FUNCTION_TOOLS))
        try:
            action, assistant = parse_tool_call(identities.decode_response(response),
                s.get('sources',[]),s.get('observations',[]),s['plan'],s.get('tool_messages',[]))
        except ToolArgumentError as exc:
            # Rejected arguments never execute. Preserve the native call/result
            # pair so the model can correct them, within the same step budget.
            assistant = exc.assistant
            call = assistant['tool_calls'][0]
            bounds = {key:datetime.fromtimestamp(s['plan'][key], now.tzinfo).isoformat()
                      if s['plan'].get(key) is not None else None for key in ('since_ts','until_ts')}
            note = f'工具未执行，请修正参数或换工具。{exc}。本轮数据边界：{json.dumps(bounds,ensure_ascii=False)}'
            result = {'tool':call['function']['name'], 'items':[], 'matched':None,
                      'complete':False, 'note':note,
                      'error':{'code':'invalid_tool_arguments','message':str(exc)}}
            s['tool_messages'] = [*s.get('tool_messages',[]), assistant,
                                  tool_result_message(call['id'], result)]
            s['observations'] = [*s.get('observations',[]), result]
            s['trace'] = [*s.get('trace',[]), {'tool':result['tool'],'tool_call_id':call['id'],
                'returned':0,'included':0,'matched':None,'complete':False,'note':note,'error':result['error']}]
            s['rounds'] = s.get('rounds',0)+1
            s['action'] = {'tool':'retry_decision'}
            return save(s,'工具参数被拒绝，正在让模型修正',phase='model')
        s['tool_messages'] = [*s.get('tool_messages',[]),assistant]
        s['tool_call_id'] = assistant['tool_calls'][0]['id']
        if action['tool']=='finish':
            s['evidence_pool'] = list(s.get('sources',[]))
            s['tool_messages'].append(tool_result_message(s['tool_call_id'],{'tool':'finish','note':'已结束检索，进入引用回答阶段。'}))
        if action['tool']=='finish' and action['source_ids']:
            # 选择原文时连带保留关联上下文，避免丢失支持接话判断的其他发送者消息。
            s['sources'], incomplete = select_evidence(s['sources'], action['source_ids'])
            s['coverage']=dict(s.get('coverage',{}),exhaustive=False,included=len(s['sources']))
            if incomplete:
                s['coverage']['context_incomplete'] = True
        # 相同动作不再次执行，防止模型在相同查询上循环消耗预算。
        if any(t.get('parameters')==action for t in s.get('trace',[])):
            s['coverage']=dict(s.get('coverage',{}),exhaustive=False)
            result = {'tool':action['tool'],'items':[],'complete':False,
                'error':{'code':'duplicate_query','retryable':True},
                'note':'相同查询已执行，未重复调用。请根据此前结果调整内容查询或核对范围；不要仅重复请求，空结果不表示全库不存在。'}
            s['tool_messages'].append(tool_result_message(s['tool_call_id'],result))
            s['observations'] = [*s.get('observations',[]),result]
            s['trace'] = [*s.get('trace',[]),{'tool':action['tool'],'parameters':action,'returned':0,
                'included':0,'complete':False,'error':result['error'],'note':result['note']}]
            s['rounds'] = s.get('rounds',0)+1
            return dict(s,action={'tool':'retry_decision'})
        return dict(s,action=action)

    def execute_node(s):
        s=dict(s)
        action=s['action']
        # Exclusions selected from validated identities remain active for this
        # turn; subsequent shared searches cannot silently reintroduce them.
        excluded = (action.get('filters') or {}).get('exclude_person_ids',[])
        if excluded:
            s['plan'] = dict(s['plan'],exclude_person_ids=list(dict.fromkeys([
                *s['plan'].get('exclude_person_ids',[]),*excluded])))
        # 执行结果分别进入证据工作集、结构化观察和原生工具回执，再回到决策节点。
        result=run_tool(s,action['tool'],lambda:execute_action(tools,s['plan'],action,s['sources']),action)
        observation={k:result[k] for k in ('tool','matched','next_offset','complete','note','error','scope_diagnostics','chats','people','windows','window_seconds') if k in result}
        s['observations']=[*s.get('observations',[]),observation]
        s['tool_messages']=[*s['tool_messages'],tool_result_message(s['tool_call_id'],result)]
        s['rounds']=s.get('rounds',0)+1
        return s

    def answer_node(s):
        s = dict(s)
        plan = s["plan"]
        # 有歧义或没有可引用原文时只返回限制说明，不调用回答模型补造结论。
        if not s["sources"]:
            s["answer_data"] = limitation("本轮未取得可引用的消息原文，尚不能回答这个问题；这不代表相关对话不存在。")
        else:
            save(s, "根据已核实的消息生成回答",phase='answer')
            # Retain whole local windows under both budgets. Never turn a
            # contextualised hit back into a dangling phrase by tail-slicing.
            context, trimmed = select_evidence(s['sources'], limit=final_message_limit, char_limit=char_budget)
            s["sources"] = context
            if trimmed:
                s["coverage"] = dict(s.get("coverage", {}), exhaustive=False, context_incomplete=True, included=len(context))
            if not context:
                s['answer_data'] = limitation('完整上下文超过本轮证据预算，未截取孤立短句作答。请缩小问题范围或提高证据文本预算。')
                return s
            grounded_question = task_instruction(s['current_task']) + '\n最新发言：'+question + "\n\n本轮工具执行记录与覆盖范围（查询条件不代表用户意图已经核实）：\n" + json.dumps({
                "platform_boundary": plan, "coverage": s.get("coverage"), "trace": model_trace(s),
                'context_windows':context_windows(context),
                'stop_reason':s.get('stop_reason',''),
                'review_feedback_to_verify_against_sources':s.get('review_feedback')}, ensure_ascii=False)
            tick = time.monotonic()
            s["answer_data"] = recorded_call('answer', lambda: answer(grounded_question, context, history), True)
            s['answer_timings'] = [*s.get('answer_timings',[]),
                {'elapsed_ms':round((time.monotonic()-tick)*1000), 'source_count':len(context)}]
            # 回答引用对应最终 context 顺序；非穷尽结果再附服务端限制说明。
            if not s.get("coverage", {}).get("exhaustive", False):
                s["answer_data"]["paragraphs"].append({"kind": "limitation", "text": f"本轮纳入 {len(context)} 条证据，属于选取的相关片段，不代表完整检查了范围内全部消息。", "source_refs": []})
        check()
        return s

    def review_node(s):
        s = dict(s)
        if not s.get('sources'):
            return dict(s,review_route='end',quality_status='no_evidence')
        save(s,'检查回答与证据是否一致',phase='review')
        tick = time.monotonic()
        fingerprint = hashlib.sha256(json.dumps({'answer':s['answer_data'],'sources':s['sources'],
            'task':s['current_task'],'coverage':s.get('coverage',{})},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        previous = next((r for r in s.get('reviews',[]) if r.get('fingerprint')==fingerprint),None)
        if previous:
            # Re-reading the identical answer and evidence adds no information.
            result = {k:previous[k] for k in ('checks','coverage','task_checks','decision','issues')}
        else:
            result = review_answer(model,question,history,s['answer_data'],s['sources'],
                                   model_trace(s),s.get('coverage',{}),now,current_task=s['current_task'])
        attempt = s.get('review_rounds',0)+1
        event = {**result,'attempt':attempt,'fingerprint':fingerprint,'reused':bool(previous),
                 'elapsed_ms':round((time.monotonic()-tick)*1000),
                 'source_count':len(s['sources']),
                 'source_ids':[source['evidence_id'] for source in s['sources']]}
        s['reviews'] = [*s.get('reviews',[]),event]
        s['review_rounds'] = attempt
        if result['decision']=='accept':
            s.update(review_route='end',quality_status='reviewed')
            return save(s,'回答检查完成（模型复核）')
        # One focused correction, then a final review. Never publish a draft
        # already flagged as wrong just because correction exhausted its budget.
        exhausted = (attempt >= 2 or (result['decision']=='retrieve' and
                     (s.get('rounds',0)>=8 or len(s.get('evidence_pool',s['sources']))>=retrieval_limit)))
        if exhausted:
            s.update(review_route='end',quality_status='unresolved')
            s['answer_data'] = limitation('回答检查仍发现未解决的问题，本轮未发布草稿结论。请查看查询过程中的检查记录，或缩小问题后重试。')
            return save(s,'回答检查未通过，已保留证据与检查记录')
        s['review_feedback'] = result
        s['quality_status'] = 'correcting'
        s['review_route'] = 'answer' if result['decision']=='revise' else 'decide'
        if s['review_route']=='decide':
            # Restore evidence discarded by finish selection before resuming
            # tools; the next answer gets a fresh, consistently numbered list.
            s['sources'] = s.get('evidence_pool') or s['sources']
        return save(s,'修正回答中的证据使用' if s['review_route']=='answer' else '重新决策：补查缺口或缩小结论')

    graph = StateGraph(AgentState)
    def durable_node(name, node):
        def run(s):
            nonlocal usage, model_metrics, transport_calls, live_state
            check()
            if s.get('platform_state') is not None:
                tools.restore(s['platform_state'])
            usage = dict(s.get('usage') or {"prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "calls": 0,
                "scope": "tool_decisions_and_reviews"})
            model_metrics = list(s.get('model_metrics', []))
            transport_calls = list(s.get('transport_calls', []))
            live_state = s
            # 回执按节点和轮次隔离；深拷贝后执行，节点成功返回才交付新的图状态。
            scope = f"{name}:{s.get('rounds',0)}" if name in ('decide','execute') else name
            scope += f":review-{s.get('review_rounds',0)}"
            with receipts.scope(scope), observe_model_transport(transport_progress):
                result = node(deepcopy(s))
            result['usage'] = dict(usage)
            result['model_metrics'] = list(model_metrics)
            result['transport_calls'] = list(transport_calls)
            return result
        return run
    for name, node in (("plan", plan_node), ("decide", decide_node),
                       ("execute", execute_node), ("answer", answer_node), ('review',review_node)):
        graph.add_node(name, durable_node(name, node))
    for a, b in ((START, 'plan'), ('execute', 'decide'), ('answer','review')):
        graph.add_edge(a, b)
    graph.add_conditional_edges('plan',lambda s:'clarify' if s['current_task'].get('clarification') else 'decide',{'clarify':END,'decide':'decide'})
    graph.add_conditional_edges('decide', lambda s: 'answer' if s['action']['tool']=='finish'
                                else 'decide' if s['action']['tool']=='retry_decision' else 'execute')
    graph.add_conditional_edges('review',lambda s:s['review_route'],{'end':END,'answer':'answer','decide':'decide'})
    # Never upload local conversations via environment-enabled LangSmith tracing.
    # 显式关闭外部追踪，避免本地聊天内容随环境配置上传。
    with tracing_context(enabled=False):
        compiled = graph.compile(checkpointer=saver)
        config = {'configurable': {'thread_id': 'qa'}, 'recursion_limit': 40}
        if resume and not compiled.get_state(config).values:
            # The first input checkpoint can be empty if planning was interrupted.
            if not saver.get_tuple(config):
                raise ValueError('没有可恢复的问答检查点')
        # 恢复用 None 续跑已存图状态；同步落盘配合调用回执，避免重放已完成请求。
        result = compiled.invoke(None if resume else {}, config, durability='sync')
    result['active_step'] = None
    return result["answer_data"], result["sources"], diagnostics(result)
