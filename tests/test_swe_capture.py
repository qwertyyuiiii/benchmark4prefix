"""Offline checks for capture fidelity, import formats, and validation."""
import json
import tempfile
import unittest
from pathlib import Path

from swe_capture import (
    Recorder, export_sessions, load_api, load_chatgpt, load_claude,
    load_messages, read_jsonl, validate,
)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        return path

    def recorder(self):
        return Recorder(self.root / 'raw.jsonl', session_id='test', agent='offline')

    def test_bundled_messages_keep_unknown_metadata(self):
        sessions = load_messages(Path(__file__).resolve().parents[1] / 'examples/messages.json')
        report = export_sessions(sessions, self.root / 'dataset')
        self.assertTrue(report['valid'])
        self.assertEqual((report['row_count'], report['user_prompts'], report['assistant_responses']), (3, 1, 1))
        self.assertTrue(all(row['timestamp'] is None and row['model'] is None for row in sessions[0].rows))

    def test_openai_round_trip_full_payload_and_call_start(self):
        recorder = self.recorder()
        payload = {'model': 'request-model', 'messages': [{'role': 'user', 'content': '请计算。'}],
                   'tools': [{'type': 'function', 'function': {'name': 'calculate', 'parameters': {'type': 'object'}}}],
                   'stream': False}
        response = {'id': 'r1', 'model': 'server-model', 'choices': [{'message': {
            'role': 'assistant', 'content': '结果。'}}], 'usage': {'prompt_tokens': 12, 'completion_tokens': 3}}
        self.assertIs(recorder.call('openai-chat', payload, lambda: response), response)
        raw = read_jsonl(recorder.path)
        self.assertEqual(raw[0]['request'], payload)
        self.assertEqual(raw[1]['response'], response)
        self.assertEqual(raw[0]['timestamp'], raw[1]['timestamp'])
        session, = load_api(recorder.path)
        self.assertEqual(session.calls[0]['model'], 'request-model')
        self.assertTrue(all(row['timestamp'] == raw[0]['timestamp'] for row in session.rows))
        report = export_sessions([session], self.root / 'dataset')
        self.assertTrue(report['valid'])
        self.assertEqual(report['warnings'], [])
        exported = read_jsonl(self.root / 'dataset/sessions.jsonl')[0]
        self.assertEqual((exported['input_tokens'], exported['output_tokens']), (12, 3))
        self.assertEqual(recorder.path.stat().st_mode & 0o777, 0o600)

    def test_anthropic_tools_and_long_results_survive(self):
        recorder = self.recorder()
        result = '完整工具结果\n' * 20000
        payload = {'model': 'anthropic-test', 'system': '系统提示', 'messages': [
            {'role': 'user', 'content': '计算'},
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 't1', 'name': 'calc', 'input': {'x': 17}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': result}]},
        ]}
        recorder.call('anthropic', payload, lambda: {
            'id': 'a1', 'content': [{'type': 'text', 'text': '完成'}],
            'usage': {'input_tokens': 10, 'output_tokens': 2}})
        session, = load_api(recorder.path)
        tool_result, = [r for r in session.rows if r['turn_type'] == 'tool_result']
        self.assertEqual(tool_result['content'], result)
        report = export_sessions([session], self.root / 'dataset')
        self.assertTrue(report['valid'])
        self.assertEqual((report['user_prompts'], report['tool_calls'], report['tool_results']), (1, 1, 1))

    def test_errors_and_incomplete_calls_remain_visible(self):
        recorder = self.recorder()
        payload = {'model': 'test', 'messages': [{'role': 'user', 'content': 'hello'}]}
        def fail():
            raise RuntimeError('offline failure')
        with self.assertRaises(RuntimeError):
            recorder.call('openai-chat', payload, fail)
        recorder.start_call('openai-chat', payload)
        session, = load_api(recorder.path)
        self.assertEqual([c['status'] for c in session.calls], ['error', 'incomplete'])
        self.assertTrue(export_sessions([session], self.root / 'dataset')['valid'])

    def test_history_dedup_keeps_every_actual_call(self):
        recorder = self.recorder()
        messages = [{'role': 'user', 'content': 'question'}]
        answer = {'role': 'assistant', 'content': 'answer'}
        for index in range(2):
            recorder.call('openai-chat', {'model': 'test', 'messages': messages},
                          lambda: {'id': f'r{index}', 'choices': [{'message': answer}]})
            messages = messages + [answer, {'role': 'user', 'content': 'next'}]
        session, = load_api(recorder.path)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(len(session.rows), 4)
        self.assertEqual(session.history_changes, 0)

    def test_chatgpt_import_selects_active_branch(self):
        def node(parent, text):
            return {'parent': parent, 'message': {'author': {'role': 'user'}, 'content': {'parts': [text]}}}
        path = self.write_json('chat.json', [{'id': 'chat', 'current_node': 'chosen', 'mapping': {
            'root': {'parent': None, 'message': None}, 'chosen': node('root', '保留'),
            'other': node('root', '不选取')}}])
        session, = load_chatgpt(path, 'chat')
        self.assertEqual([r['content'] for r in session.rows], ['保留'])
        self.assertTrue(export_sessions([session], self.root / 'dataset')['valid'])

    def test_claude_import_preserves_metadata_events(self):
        path = self.root / 'claude.jsonl'
        records = [
            {'sessionId': 'c1', 'type': 'user', 'message': {'content': 'hello'}},
            {'sessionId': 'c1', 'type': 'summary', 'summary': 'compressed'},
            {'sessionId': 'c1', 'type': 'assistant', 'message': {'id': 'a1', 'content': 'world'}},
        ]
        path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        session, = load_claude(path)
        self.assertEqual([r['turn_type'] for r in session.rows], ['user_prompt', 'summary', 'assistant_response'])
        self.assertTrue(export_sessions([session], self.root / 'dataset')['valid'])

    def test_validation_detects_transcript_tampering_and_prevents_overwrite(self):
        sessions = load_messages(self.write_json('messages.json', {'session_id': '../outside', 'messages': [
            {'role': 'user', 'content': 'hello'}]}))
        output = self.root / 'dataset'
        self.assertTrue(export_sessions(sessions, output)['valid'])
        transcript, = (output / 'transcripts').glob('*.jsonl')
        self.assertTrue(transcript.resolve().is_relative_to(output.resolve()))
        with self.assertRaisesRegex(ValueError, 'must be empty'):
            export_sessions(sessions, output)
        with transcript.open('a', encoding='utf-8') as stream:
            stream.write('{}\n')
        self.assertIn('raw transcript checksum mismatch', validate(output)['errors'])


if __name__ == '__main__':
    unittest.main()
