"""Optional real-framework regression test; no network or API key required."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('smolagents') and importlib.util.find_spec('openai'),
                     'Install requirements-agent-test.txt for the smolagents integration test')
class SmolagentsIntegrationTests(unittest.TestCase):
    def test_tool_loop_capture_and_export(self):
        from examples.smolagents_capture import run
        from swe_capture import read_jsonl
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            report = run(output, offline=True)
            self.assertTrue(all(report['checks'].values()))
            self.assertEqual(report['call_count'], 2)
            self.assertEqual(report['mode'], 'offline_scripted')
            quality, = json.loads((output / 'dataset/capture_quality.json').read_text())
            self.assertFalse(quality['provenance']['actual_request_payloads'])
            self.assertTrue(quality['provenance']['synthetic_model_response'])
            starts = [r for r in read_jsonl(output / 'raw.jsonl') if r['event'] == 'request_start']
            self.assertTrue(all('extra_body' not in r['request'] for r in starts))
            self.assertTrue(all(r['request']['thinking'] == {'type': 'disabled'} for r in starts))


if __name__ == '__main__':
    unittest.main()
