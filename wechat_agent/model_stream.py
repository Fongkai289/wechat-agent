"""Assemble streamed Chat Completions without exposing unvalidated JSON/tools.

Progress contains timing/counts only. A partial stream is never a completed
response or a durable receipt, and is never retried here.
"""
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar

_observer = ContextVar('model_transport_observer', default=None)


@contextmanager
def observe_model_transport(callback):
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


def report_transport(event):
    callback = _observer.get()
    if callback:
        callback(dict(event))


def read_chat_stream(response, metrics, check):
    start = time.monotonic()
    base_ms = metrics.get('headers_ms', 0)
    choices, usage, fields = {}, {}, {}
    done, first_event, first_content, output_chars = False, None, None, 0
    last_report = 0.0

    def elapsed():
        return base_ms + round((time.monotonic()-start)*1000)

    def update(status, force=False):
        nonlocal last_report
        tick = time.monotonic()
        metrics.update(status=status, first_event_ms=first_event,
                       first_content_ms=first_content, output_chars=output_chars,
                       elapsed_ms=elapsed())
        if force or tick-last_report >= 1:
            report_transport(metrics)
            last_report = tick

    data_lines = []
    for raw in response:
        check()
        line = raw.decode('utf-8').rstrip('\r\n')
        if line:
            if line.startswith('data:'):
                data_lines.append(line[5:].lstrip(' '))
            continue
        if not data_lines:
            continue  # SSE comments/keep-alives are not model output.
        data = '\n'.join(data_lines)
        data_lines.clear()
        if data == '[DONE]':
            done = True
            break
        chunk = json.loads(data)
        if chunk.get('error'):
            raise RuntimeError('模型流式响应返回错误：'+str(chunk['error'].get('message','未知错误'))[:300])
        if first_event is None:
            first_event = elapsed()
        for key in ('id','model','created','system_fingerprint'):
            if key in chunk: fields[key] = chunk[key]
        if chunk.get('usage') is not None: usage = chunk['usage']
        for part in chunk.get('choices', []):
            index = part['index']
            choice = choices.setdefault(index, {'index':index,'finish_reason':None,
                'message':{'role':'assistant','content':''}, '_tools':{}})
            delta = part.get('delta') or {}
            message = choice['message']
            if delta.get('role'): message['role'] = delta['role']
            for key in ('content','refusal'):
                text = delta.get(key)
                if text:
                    if not isinstance(text,str): raise ValueError('模型流式文本格式无效')
                    message[key] = message.get(key,'') + text
                    output_chars += len(text)
                    if first_content is None: first_content = elapsed()
            for tool in delta.get('tool_calls') or []:
                target = choice['_tools'].setdefault(tool['index'], {'id':'','type':'function',
                    'function':{'name':'','arguments':''}})
                if tool.get('id'): target['id'] += tool['id']
                if tool.get('type'): target['type'] = tool['type']
                for key in ('name','arguments'):
                    text = (tool.get('function') or {}).get(key) or ''
                    target['function'][key] += text
                    output_chars += len(text)
                    if text and first_content is None: first_content = elapsed()
            if part.get('finish_reason') is not None:
                choice['finish_reason'] = part['finish_reason']
        update('receiving')
    check()
    if not done or not choices or any(c['finish_reason'] is None for c in choices.values()):
        raise RuntimeError('模型流式响应中断，未收到完整结束标记；未自动重试')
    for choice in choices.values():
        tools = choice.pop('_tools')
        if tools:
            choice['message']['tool_calls'] = [tools[i] for i in sorted(tools)]
    metrics['total_ms'] = elapsed()
    metrics.update({key:usage.get(key) for key in ('prompt_tokens','completion_tokens','total_tokens')})
    update('completed', True)
    return {**fields, 'choices':[choices[i] for i in sorted(choices)], 'usage':usage,
            '_transport':dict(metrics)}
