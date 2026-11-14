"""One evidence catalog per decision; links use compact local references."""
import json

from .evidence_context import ordered_source_groups


def decision_evidence(sources, text_budget=30000, tool_messages=()):
    catalog = [{**{k:s.get(k) for k in ('evidence_id','chat_id',
                'sender','sender_username','time','evidence_role')}, 'reference':i+1} for i,s in enumerate(sources)]
    chats = list({s.get('chat_id'):{k:s.get(k) for k in ('chat_id','chat_title','chat_type')} for s in sources}.values())
    previews = []
    # Share the preview budget across the entire catalog. Old tool responses
    # must not reinsert hundreds of full texts after this budget is applied.
    # Full originals remain in the evidence store for context tools and answers.
    for i,source in enumerate(sources):
        if text_budget <= 0: break
        text = str(source.get('text') or '')
        allowance = min(1500, max(1,text_budget // (len(sources)-i)))
        excerpt = text[:allowance]
        text_budget -= len(excerpt)
        previews.append({'reference':i+1,'text':excerpt,'preview_truncated':len(excerpt)<len(text)})
    # Decision needs a compact chronological view, not every overlapping
    # anchor's neighbour graph. Originals and exact links stay in the host.
    blocks = [{'chat_id':group[0][1].get('chat_id'), 'ordered_refs':[ref for ref,_ in group]}
              for group in ordered_source_groups(sources)]
    return {'source_catalog':catalog, 'chats':chats, 'evidence':previews, 'chronological_blocks':blocks,
            'evidence_format':'reference 对应 source_catalog；工具的 source_ids 使用目录中的 evidence_id。'
                              'chronological_blocks仅按会话和时间分组，不证明同一话题或回复。'
                              '原文预览有长度预算，未展示不代表不存在；可调用上下文工具读取。'}


def compact_tool_history(messages):
    """Keep native call/result pairing, but do not resend old evidence graphs."""
    result = []
    for message in messages:
        if message.get('role') != 'tool':
            result.append(message)
            continue
        data = json.loads(message['content'])
        compact = {k:data[k] for k in ('tool','matched','returned','next_offset','complete',
                   'note','error','scope_diagnostics','anchor_count','context_count') if k in data}
        if 'items' in data:
            compact['returned'] = len(data['items'])
        compact['evidence_view'] = '人物与会话见 observations；消息与上下文见本轮 source_catalog、evidence 和 context_windows。'
        result.append({**message,'content':json.dumps(compact,ensure_ascii=False)})
    return result
