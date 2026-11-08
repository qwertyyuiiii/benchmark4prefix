"""Run a real smolagents tool loop and capture its non-streaming API calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swe_capture import Recorder, export_sessions, load_api, message_text, read_jsonl
from smolagents import OpenAIServerModel, ToolCallingAgent, tool
from smolagents.models import tool_role_conversions


class CapturedModel(OpenAIServerModel):
    """Pinned to smolagents 1.22.0's explicit message conversion rules.

    One model instance per sequential agent. Capture the final SDK body, not
    client configuration, keys, headers, or the framework's pre-conversion input.
    SDK retries are disabled so one recorded call equals one HTTP attempt.
    """
    def __init__(self, recorder, **kwargs):
        self.recorder = recorder
        self.agent = None
        self._annotations = []
        super().__init__(**kwargs)
        invoke = self.client.chat.completions.create

        def captured_create(**payload):
            if payload.get('stream') or any(k in payload for k in ('extra_headers', 'extra_query')):
                raise ValueError('Only non-streaming body parameters are supported')
            body = {k: v for k, v in payload.items() if k != 'extra_body'}
            body.update(payload.get('extra_body') or {})
            return recorder.call('openai-chat', body, lambda: invoke(**payload),
                                 replay_metadata={'message_annotations': self._annotations})

        self.client.chat.completions.create = captured_create

    def _prepare_completion_kwargs(self, messages, **kwargs):
        payload = super()._prepare_completion_kwargs(messages=messages, **kwargs)
        conversions = self.custom_role_conversions or tool_role_conversions
        groups = []
        for source in messages:
            role = source.get('role') if isinstance(source, dict) else source.role
            converted = conversions.get(role, role)
            if groups and groups[-1][0] == converted:
                groups[-1][1].append(role)
            else:
                groups.append((converted, [role]))
        if len(groups) != len(payload['messages']):
            raise ValueError('Unsupported smolagents message grouping')

        # Match against structured framework memory, never infer tool semantics
        # solely from an untrusted user's text beginning with "Observation:".
        results = {}
        if self.agent is not None:
            for step in self.agent.memory.steps:
                calls = getattr(step, 'tool_calls', None) or []
                if len(calls) == 1:
                    for msg in step.to_messages():
                        if msg.role == 'tool-response':
                            text = message_text({'content': msg.content})
                            results.setdefault(text, []).append(calls[0])

        annotations = []
        for index, ((_, source_roles), msg) in enumerate(zip(groups, payload['messages'])):
            text = message_text(msg)
            annotation = {'index': index, 'content_sha256': hashlib.sha256(text.encode()).hexdigest()}
            if set(source_roles) == {'tool-response'}:
                matches = results.get(text, [])
                if len(matches) == 1:
                    annotation.update(kind='tool_result', tool_call_id=matches[0].id,
                                      tool_name=matches[0].name)
                else:
                    # Mixed/batched/error observations cannot safely be linked
                    # to a single tool. Keep them as explicit framework events.
                    annotation['kind'] = 'system_injected'
            elif 'tool-call' in source_roles:
                annotation['kind'] = 'history_rendering'
            else:
                continue
            annotations.append(annotation)
        self._annotations = annotations
        return payload


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers exactly.

    Args:
        a: First integer.
        b: Second integer.
    """
    return a * b


def run(output_dir, *, offline=False, model_id='deepseek-flash', api_base='https://api.deepseek.com'):
    output = Path(output_dir)
    # Refuse reuse before sending any billable request.
    output.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(output / 'raw.jsonl', session_id='smolagents-' + uuid.uuid4().hex,
                        agent='smolagents-1.22.0', repo_id='qwertyyuiiii/benchmark4prefix')
    key = 'offline-placeholder' if offline else os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        raise ValueError('Set DEEPSEEK_API_KEY in the process environment')
    model = CapturedModel(
        recorder, model_id='offline-scripted-model' if offline else model_id,
        api_base=api_base, api_key=key, flatten_messages_as_text=True,
        client_kwargs={'max_retries': 0, 'timeout': 90},
        temperature=0, max_tokens=512, extra_body={'thinking': {'type': 'disabled'}},
    )
    if offline:
        from openai.types.chat import ChatCompletion
        count = 0
        # Replace the underlying SDK call, then install the same recorder hook.
        # This is an explicit synthetic response fixture, not model inference.
        def scripted_create(**payload):
            nonlocal count
            count += 1
            if count > 2:
                raise AssertionError('Unexpected additional model call')
            name, arguments = ('multiply', {'a': 17, 'b': 23}) if count == 1 else ('final_answer', {'answer': '391'})
            response = ChatCompletion.model_validate({
                'id': f'offline-{count}', 'object': 'chat.completion', 'created': 0,
                'model': 'offline-scripted-model', 'choices': [{'index': 0, 'finish_reason': 'tool_calls',
                    'message': {'role': 'assistant', 'content': None, 'tool_calls': [{
                        'id': f'offline-tool-{count}', 'type': 'function',
                        'function': {'name': name, 'arguments': json.dumps(arguments)}}]}}],
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            })
            body = {k: v for k, v in payload.items() if k != 'extra_body'}
            body.update(payload.get('extra_body') or {})
            return recorder.call('openai-chat', body, lambda: response,
                                 replay_metadata={'message_annotations': model._annotations,
                                                  'synthetic_model_response': True})
        model.client.chat.completions.create = scripted_create

    agent = ToolCallingAgent(tools=[multiply], model=model, max_steps=3,
                             verbosity_level=0, stream_outputs=False)
    model.agent = agent
    answer = agent.run('Use multiply exactly once to compute 17 * 23. Then call final_answer with the result. Do not compute it yourself.')
    sessions = load_api(recorder.path)
    if offline:
        for session in sessions:
            session.provenance.update(actual_request_payloads=False,
                                      synthetic_model_response=True,
                                      source_kind='offline_agent_fixture')
    validation = export_sessions(sessions, output / 'dataset')
    rows = read_jsonl(output / 'dataset/conversations.jsonl')
    calls = read_jsonl(output / 'dataset/calls.jsonl')
    raw = read_jsonl(recorder.path)
    checks = {
        'correct_answer': '391' in str(answer),
        'valid_dataset': validation['valid'],
        'all_calls_completed': bool(calls) and all(c['status'] == 'completed' for c in calls),
        'one_human_prompt': validation['user_prompts'] == 1,
        'multiply_result_captured': any(r['turn_type'] == 'tool_result' and r['tool_name'] == 'multiply' and '391' in r['content'] for r in rows),
        'tool_schemas_captured': all(r['request'].get('tools') for r in raw if r['event'] == 'request_start'),
        'no_truncation': all(not a['content_truncated'] for a in read_jsonl(output / 'dataset/capture_annotations.jsonl')),
    }
    report = {'agent': 'smolagents-1.22.0', 'mode': 'offline_scripted' if offline else 'live_api',
              'model': calls[0]['model'] if calls else None, 'answer': str(answer),
              'call_count': len(calls), 'checks': checks, 'validation': validation}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise RuntimeError('Agent capture checks failed; see report.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='A new directory, never reused')
    parser.add_argument('--offline', action='store_true', help='Use explicit scripted response fixtures')
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--api-base', default='https://api.deepseek.com')
    args = parser.parse_args()
    run(args.output_dir, offline=args.offline, model_id=args.model, api_base=args.api_base)


if __name__ == '__main__':
    main()
