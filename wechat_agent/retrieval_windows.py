"""Rank bounded conversation windows, then expand intact original messages."""
import hashlib
import json


def message_key(row):
    return tuple(str(row.get(k) or '') for k in
                 ('chat_id','source_db','source_table','local_id','server_id','timestamp'))


def rerank_windows(candidates, semantic_groups, max_chars=6000, max_messages=48):
    by_key = {message_key(row):row for row in candidates}
    assigned, windows = set(), []

    def add_group(rows):
        batch, length = [], 0
        def flush():
            if not batch: return
            text='\n'.join(line for _,line in batch)
            members=[row for row,_ in batch]
            digest=hashlib.sha256(json.dumps([message_key(r) for r in members]).encode()).hexdigest()[:24]
            windows.append({**members[0], 'local_id':'window:'+digest,
                'text':text, 'sender':'会话窗口（各条发送者见原文）',
                'retrieval_context':'', 'rerank_unit':'conversation_window',
                'retrieval_members':members, 'fusion_score':max(r.get('fusion_score',0) for r in members)})
        for row in rows:
            key=message_key(row)
            if key in assigned: continue
            line=f"{row.get('time','')} · {row.get('sender','')} · {row.get('type','')}：{row.get('text','')}"
            if batch and (row.get('chat_id')!=batch[-1][0].get('chat_id') or
                          length+len(line)+1>max_chars or len(batch)>=max_messages):
                flush(); batch, length = [], 0
            batch.append((row,line)); length+=len(line)+1; assigned.add(key)
        flush()

    # A semantic block is already a conversation unit. Keep all scoped members
    # together; a mixed-scope block has already been filtered by the caller.
    for keys in semantic_groups:
        add_group([by_key[key] for key in keys if key in by_key])
    # Lexical hits inside semantic blocks share the same scoring request.
    # Nearby remaining lexical hits can share a window; this does not assert
    # that neighbouring speakers are replying to one another.
    remaining=sorted((r for r in candidates if message_key(r) not in assigned),
                     key=lambda r:(str(r.get('chat_id','')),r.get('timestamp',0),str(r.get('local_id',''))))
    group=[]
    for row in remaining:
        if group and (row.get('chat_id')!=group[-1].get('chat_id') or
                      row.get('timestamp',0)-group[-1].get('timestamp',0)>1800):
            add_group(group); group=[]
        group.append(row)
    add_group(group)
    return sorted(windows,key=lambda r:-r.get('fusion_score',0))


def expand_ranked_windows(windows, target, diversify_by_chat=False):
    """A message target is soft at window boundaries: never drop the key tail."""
    rows, seen = [], set()
    if diversify_by_chat:
        # Keep the best window per chat first, without changing model scores.
        # Preserve at least two complete windows when available, even if the
        # top window crosses the message target. Further windows use that target.
        distinct, remainder, chats = [], [], set()
        for window in windows:
            chat = window.get('chat_id')
            if chat in chats:
                remainder.append(window)
            else:
                distinct.append(window); chats.add(chat)
        windows = distinct + remainder
        minimum_windows = min(2,len(distinct))
    else:
        minimum_windows = 1
    for index,window in enumerate(windows):
        for member in window['retrieval_members']:
            key=message_key(member)
            if key in seen: continue
            seen.add(key)
            rows.append({**member, 'rerank_score':window.get('rerank_score'),
                         'rerank_reason':window.get('rerank_reason',''),
                         'ranked_window_id':window['local_id']})
        if len(rows)>=target and index+1>=minimum_windows: break
    return rows
