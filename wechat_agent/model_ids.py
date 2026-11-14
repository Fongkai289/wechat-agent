"""Typed model-facing handles; database identities stay on the host side."""
import copy
import hashlib
import json
from .model_output import strict_json


ID_FIELDS = {'chat_id': 'chat', 'chat_ids': 'chat', 'person_id': 'person',
             'person_ids': 'person', 'sender_username': 'person',
             'evidence_id': 'evidence', 'source_ids': 'evidence',
             'context_ids': 'evidence', 'context_for': 'evidence', 'exclude_person_ids':'person'}
INSTRUCTIONS = ('工具 ID 分为 chat_（会话）、person_（人物）、evidence_（消息证据），不可互换。'
                '只使用本轮已返回的对应类型 ID；微信账号或历史正文中的原始 ID 不是工具参数。'
                '需要指定会话或人物ID时，分别用list_chats/read_chats或find_people查明身份。'
                '搜索消息内容可直接调用search，不必先定位人物；find_people仅按姓名或账号查找。'
                'query_messages 的身份条件使用 SQL 占位符，在 params 传对应类型 ID，不在 SQL 中拼接 ID。')


def model_id(kind, raw):
    return kind + '_' + hashlib.sha256(str(raw).encode()).hexdigest()[:24]


class ModelIDs:
    def __init__(self, sources, observations):
        self.known = {kind: {} for kind in ('chat', 'person', 'evidence')}
        self.directory = []
        for source in sources:
            self.add('chat', source.get('chat_id'), source.get('chat_title'))
            self.add('person', source.get('sender_username'), source.get('sender'))
            self.add('evidence', source.get('evidence_id'))
        for observation in observations:
            for chat in observation.get('chats', []):
                self.add('chat', chat.get('chat_id'), chat.get('chat_title'))
                for sender in chat.get('senders', []):
                    self.add('person', sender.get('sender_username'), sender.get('name'))
            for person in observation.get('people', []):
                self.add('person', person.get('id'), person.get('name'))

    def add(self, kind, raw, label=None):
        if not isinstance(raw, str) or not raw:
            return
        handle = model_id(kind, raw)
        if handle in self.known[kind]:
            if self.known[kind][handle] != raw:
                raise ValueError('工具 ID 映射冲突')
            return
        self.known[kind][handle] = raw
        self.directory.append({'type': kind, 'id': handle, 'name': label or ''})

    def encode(self, value, parent=''):
        if isinstance(value, list):
            return [self.encode(item, parent) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            kind = ID_FIELDS.get(key)
            if key == 'id' and parent in ('people', 'senders'):
                kind = 'person'
            if key == 'id' and parent == 'chats':
                kind = 'chat'
            if kind:
                def handle(raw):
                    if not isinstance(raw, str) or not raw:
                        return raw
                    return raw if raw in self.known[kind] else model_id(kind, raw)
                result[key] = [handle(v) for v in item] if isinstance(item, list) else handle(item)
            else:
                result[key] = self.encode(item, key)
        return result

    def definitions(self, definitions):
        result = copy.deepcopy(definitions)
        # Stable typed schemas avoid repeating the entire identity catalog in
        # every tool definition. Exact membership is still checked on decode.
        for tool in result:
            props = tool['function']['parameters']['properties']
            for field, spec in props.items():
                kind = ID_FIELDS.get(field)
                if not kind:
                    continue
                ids = sorted(self.known[kind])
                allowed = {'type': 'string', 'pattern': '^' + kind + '_[0-9a-f]{24}$'}
                if spec.get('type') == 'array':
                    spec['items'] = allowed
                    if not ids:
                        spec['maxItems'] = 0
                else:
                    props[field] = {**allowed, 'description': spec.get('description', '')}
        return result

    def messages(self, messages, include_directory=True):
        result = copy.deepcopy(messages)
        for message in result:
            content = message.get('content')
            if isinstance(content, str):
                try:
                    data = json.loads(content)
                except ValueError:
                    pass  # Never rewrite user prose or message contents by substring.
                else:
                    message['content'] = json.dumps(self.encode(data), ensure_ascii=False)
            for call in message.get('tool_calls', []):
                function = call['function']
                function['arguments'] = json.dumps(self.encode(json.loads(function['arguments'])), ensure_ascii=False)
        # The directory also supports Skills that construct free-text evidence prompts.
        directory = '\n' + json.dumps({'tool_identity_directory': self.directory}, ensure_ascii=False) if include_directory else ''
        result.insert(0, {'role':'system', 'content':INSTRUCTIONS + directory})
        return result

    def decode_arguments(self, args):
        if not isinstance(args, dict):
            raise ValueError('工具参数必须是对象')
        result = copy.deepcopy(args)
        if isinstance(args.get('filters'),dict):
            result['filters'] = self.decode_arguments(args['filters'])
        for field, value in args.items():
            kind = ID_FIELDS.get(field)
            if not kind:
                continue
            def decode(handle):
                if not isinstance(handle, str) or handle not in self.known[kind]:
                    raise ValueError('工具参数 ' + field + ' 需要已返回的 ' + kind +
                                     '_ ID，不能使用其他类型或原始微信 ID')
                return self.known[kind][handle]
            result[field] = [decode(v) for v in value] if isinstance(value, list) else decode(value)
        # SQL remains a code interface; translate explicit bound handles only.
        if isinstance(args.get('params'), list):
            all_ids = {k: v for mapping in self.known.values() for k, v in mapping.items()}
            result['params'] = [all_ids.get(v, v) if isinstance(v, str) else v for v in args['params']]
        return result

    def decode_response(self, payload):
        result = copy.deepcopy(payload)
        if not isinstance(result, dict):
            return result
        choices = result.get('choices')
        if not isinstance(choices, list):
            return result
        for choice in choices:
            if not isinstance(choice, dict) or not isinstance(choice.get('message'), dict):
                continue
            calls = choice['message'].get('tool_calls')
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
                    continue
                function = call['function']
                if not isinstance(function.get('arguments'), str):
                    continue
                function['arguments'] = json.dumps(self.decode_arguments(strict_json(function['arguments'])), ensure_ascii=False)
        return result
