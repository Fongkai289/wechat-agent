"""Persist the user's evolving request independently of retrieved evidence."""
import hashlib
import json
from copy import deepcopy

from .model_output import STRING, object_schema, structured_format, parse_structured_completion

ORIGIN = {'source_turn': {'type':'integer','minimum':1}, 'source_quote': STRING}
TASK_FORMAT = structured_format('current_conversation_task', object_schema({
    'request': STRING, 'expected_result': STRING, 'clarification': STRING,
    'add_requirements': {'type':'array','items':object_schema({'text':STRING,
        'importance':{'type':'string','enum':['constraint','priority','reference']}, **ORIGIN})},
    'remove_requirements': {'type':'array','items':object_schema({'id':STRING, **ORIGIN})},
}))


def task_view(task):
    """The same concise contract goes to search, generation, review and the UI."""
    return {k:deepcopy(task[k]) for k in ('request','expected_result','requirements','clarification') if k in task}


def task_inputs(question, history):
    previous, position = None, -1
    for i, message in enumerate(history):
        task = (message.get('retrieval') or {}).get('current_task')
        if isinstance(task,dict) and task.get('version') in (1,2) and isinstance(task.get('user_turns'),list):
            previous, position = task, i
    turns = list(previous['user_turns']) if previous else []
    turns.extend(str(m.get('content') or '') for m in history[position+1:] if m.get('role')=='user')
    turns.append(question)
    return previous, turns


def apply_task_update(update, previous, turns):
    """Retain old requirements unless removal cites a newer user instruction.

    Quotes verify provenance, not meaning: semantic interpretation is still
    fallible and is checked again by the answer reviewer.
    """
    previous = previous or {}
    requirements = deepcopy(previous.get('requirements') or [])
    old_count = len(previous.get('user_turns') or [])
    def origin(row, removing=False):
        n, quote = row['source_turn'], row['source_quote']
        if not 1 <= n <= len(turns) or not quote.strip() or quote not in turns[n-1]:
            raise ValueError('任务要求的来源必须引用实际用户原话')
        if removing and n <= old_count:
            raise ValueError('撤销旧要求必须依据新的用户发言')
    known = {r['id'] for r in requirements}
    removed = set()
    for row in update['remove_requirements']:
        origin(row, True)
        if row['id'] not in known or row['id'] in removed:
            raise ValueError('只能撤销现有且未重复的任务要求')
        removed.add(row['id'])
    requirements = [r for r in requirements if r['id'] not in removed]
    for row in update['add_requirements']:
        origin(row)
        if not row['text'].strip(): raise ValueError('任务要求不能为空')
        if any(r['text']==row['text'] for r in requirements): continue
        digest = hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:16]
        requirements.append({'id':'requirement_'+digest, **row})
    if not update['request'].strip() or not update['expected_result'].strip():
        raise ValueError('当前任务与预期结果不能为空')
    return {'version':2, 'request':update['request'], 'expected_result':update['expected_result'],
            'clarification':update['clarification'], 'requirements':requirements,
            'user_turns':turns, 'changes':{'added':update['add_requirements'],
                                        'removed':update['remove_requirements']}}


def resolve_task(model, question, history, now):
    previous, turns = task_inputs(question,history)
    # Legacy tasks treated every suggestion as mandatory. Reinterpret their
    # original user ledger once instead of perpetuating that stronger wording.
    if previous and previous.get('version') == 1:
        previous = None
    # A standalone request is already the task. No extra API call is needed.
    if len(turns)==1:
        return {'version':2,'request':question,'expected_result':'回答用户请求并提供相应证据',
                'clarification':'','requirements':[],'user_turns':turns,'changes':{'added':[],'removed':[]}}
    material = {'current_time':now.isoformat(), 'latest_message':question,
                'previous_task':task_view(previous or {}),
                'user_turns':[{'turn':i+1,'text':text} for i,text in enumerate(turns)],
                'recent_assistant_context':[str(m.get('content') or '')[:2000] for m in history
                    if m.get('role')=='assistant' and not any(m.get(k) for k in ('error','pending','stopped'))][-4:]}
    messages=[{'role':'system','content':
        '将用户最新消息与历史合并为本轮完整任务，不制定检索步骤、不回答事实问题。'
        '用户的追问、修改标准和索要结论通常延续原任务。保留仍有效的要求，'
        '只有用户明确修改、撤销或换话题才移除不适用要求；新一句没提到不代表撤销。'
        'request简短重述完整请求，expected_result用一句话概括用户实际要的结果。'
        '只整理已有意图，不扩写分析方案，不自行添加指标、权重、数量、时间范围或交付项目。'
        '用户没有明确改变的时间表达保持原意，不擅自换算成其他时间段。requirements由平台合并，'
        'add_requirements补充尚未记录的用户要求，remove_requirements用现有id表示撤销。'
        'importance区分constraint（明确边界或排除）、priority（用户强调的主要判断依据）、reference（可参考因素）。'
        '参考因素缺资料可以说明局限，不升级为必须满足的条件；结合后续发言更新重点。'
        '每项变更的source_turn与source_quote必须对应用户原话；助手答案不是用户要求。'
        'request和expected_result不得与保留要求冲突。无法确定任务指向时填写简短clarification，否则空字符串。'
        '要求使用通用自然语言描述，不推断固定业务类别、人物ID或用户未要求的范围。'},
        {'role':'user','content':json.dumps(material,ensure_ascii=False)}]
    update = parse_structured_completion(model(messages,TASK_FORMAT),TASK_FORMAT)
    return apply_task_update(update,previous,turns)


def task_instruction(task):
    return '本轮共同任务（constraint必须遵守；priority是判断重点；reference仅作补充，缺失可说明局限）：\n'+json.dumps(task_view(task),ensure_ascii=False)
