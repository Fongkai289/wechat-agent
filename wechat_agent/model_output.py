"""Strict JSON output helpers shared by model-facing modules."""
import json
import math
from copy import deepcopy
from jsonschema import Draft202012Validator, ValidationError


STRING = {"type": "string"}


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def structured_format(name, schema):
    return {'type':'json_schema','json_schema':{'name':name,'strict':True,'schema':deepcopy(schema)}}


TEXT_FORMAT = structured_format('platform_text', object_schema({'text': STRING}))


def output_contract(response_format=None, tools=None, tool_choice='auto'):
    """Every model request has strict tools, a strict output schema, or both."""
    if tools is not None:
        if not isinstance(tools, list) or not tools:
            raise ValueError('模型工具列表不能为空')
        for tool in tools:
            function = tool.get('function', {}) if isinstance(tool, dict) else {}
            if not isinstance(tool, dict) or not isinstance(function, dict) or tool.get('type') != 'function' or function.get('strict') is not True:
                raise ValueError('模型工具必须声明 strict=true')
            Draft202012Validator.check_schema(function.get('parameters'))
    if response_format is None:
        return None if tools and tool_choice == 'required' else deepcopy(TEXT_FORMAT)
    spec = response_format.get('json_schema', {}) if isinstance(response_format, dict) else {}
    if not isinstance(response_format, dict) or not isinstance(spec, dict) or response_format.get('type') != 'json_schema' or spec.get('strict') is not True:
        raise ValueError('模型输出必须使用 strict JSON Schema，不能降级为自由文本或 JSON mode')
    Draft202012Validator.check_schema(spec.get('schema'))
    return deepcopy(response_format)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('模型 JSON 包含重复字段：' + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('模型 JSON 包含非有限数值')
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('模型 JSON 包含非有限数值')
        return result
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)


def validate_structured_data(value, schema):
    try:
        Draft202012Validator(schema).validate(value)
    except ValidationError as exc:
        # Report field paths, never dump chat content or the whole failed instance.
        path = '.'.join(str(p) for p in exc.absolute_path) or '$'
        if exc.validator == 'required' and isinstance(exc.instance, dict):
            missing = [k for k in exc.validator_value if k not in exc.instance]
            detail = '缺少字段 ' + ', '.join(missing[:20])
        else:
            detail = '不满足 ' + str(exc.validator)
        raise ValueError('结构化输出 ' + path + '：' + detail) from None
    return value


def parse_structured_completion(payload, response_format):
    data = parse_json_completion(payload)
    return validate_structured_data(data, response_format['json_schema']['schema'])


def parse_json_completion(payload):
    choices = payload.get("choices") or []
    if not choices or choices[0].get("finish_reason") != "stop":
        raise ValueError("模型输出未完整生成，请重试")
    message = choices[0].get("message") or {}
    if message.get("refusal"):
        raise ValueError("模型拒绝了本次请求")
    data = strict_json(message.get("content") or "")
    if not isinstance(data, dict):
        raise ValueError("模型输出格式无效")
    return data
