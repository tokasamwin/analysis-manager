import json
import unittest
from unittest.mock import patch

from analysisManager.submitter import submitter


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError('not JSON')
        return self._payload


def payload(run_path='runs/01'):
    return {'config': {}, 'run_path': run_path, 'kwargs': {}}


class SubmitterTests(unittest.TestCase):
    def test_queued_then_404_reports_server_restart(self):
        worker = submitter(baseURL='http://server/api/jobs', poll=0)
        with patch('analysisManager.submitter.requests.post') as post:
            post.return_value = FakeResponse(202, {'status': 'queued'})
            with patch('analysisManager.submitter.requests.get') as get:
                get.return_value = FakeResponse(404, {'detail': 'not found'})
                with self.assertRaises(RuntimeError) as context:
                    worker(payload())
        message = str(context.exception)
        self.assertIn('HTTP 404', message)
        self.assertIn('crashed or restarted', message)
        self.assertIsNone(worker.currentSubmission)

    def test_submission_http_error_includes_status_and_body(self):
        worker = submitter(baseURL='http://server/api/jobs')
        with patch('analysisManager.submitter.requests.post') as post:
            post.return_value = FakeResponse(422, {'detail': 'bad run_path'})
            with self.assertRaises(RuntimeError) as context:
                worker(payload())
        message = str(context.exception)
        self.assertIn('HTTP 422', message)
        self.assertIn('bad run_path', message)

    def test_submission_invalid_json_raises(self):
        worker = submitter(baseURL='http://server/api/jobs')
        with patch('analysisManager.submitter.requests.post') as post:
            post.return_value = FakeResponse(200, None, text='<html>oops</html>')
            with self.assertRaises(RuntimeError) as context:
                worker(payload())
        self.assertIn('invalid JSON', str(context.exception))

    def test_submission_missing_status_raises(self):
        worker = submitter(baseURL='http://server/api/jobs')
        with patch('analysisManager.submitter.requests.post') as post:
            post.return_value = FakeResponse(200, {'detail': 'nope'})
            with self.assertRaises(RuntimeError) as context:
                worker(payload())
        self.assertIn('unexpected response', str(context.exception))

    def test_poll_http_error_raises(self):
        worker = submitter(baseURL='http://server/api/jobs')
        worker.currentSubmission = 'runs/01'
        with patch('analysisManager.submitter.requests.get') as get:
            get.return_value = FakeResponse(500, {'detail': 'boom'})
            with self.assertRaises(RuntimeError) as context:
                worker._wait()
        self.assertIn('HTTP 500', str(context.exception))

    def test_poll_unknown_status_raises_instead_of_looping(self):
        worker = submitter(baseURL='http://server/api/jobs')
        worker.currentSubmission = 'runs/01'
        with patch('analysisManager.submitter.requests.get') as get:
            get.return_value = FakeResponse(200, {'detail': 'no status here'})
            with self.assertRaises(RuntimeError) as context:
                worker._wait()
        self.assertIn('unexpected job status', str(context.exception))

    def test_running_then_complete_returns_result(self):
        worker = submitter(baseURL='http://server/api/jobs', poll=0)
        responses = [
            FakeResponse(200, {'status': 'running'}),
            FakeResponse(200, {'status': 'complete', 'result': {'k_eff': 0.9}}),
        ]
        with patch('analysisManager.submitter.requests.post') as post:
            post.return_value = FakeResponse(202, {'status': 'queued'})
            with patch('analysisManager.submitter.requests.get', side_effect=responses):
                result = worker(payload())
        self.assertEqual(result, {'k_eff': 0.9})
        self.assertIsNone(worker.currentSubmission)

    def test_failed_status_raises_with_error(self):
        worker = submitter(baseURL='http://server/api/jobs', poll=0)
        with patch('analysisManager.submitter.requests.post') as post:
            post.return_value = FakeResponse(202, {'status': 'queued'})
            with patch('analysisManager.submitter.requests.get') as get:
                get.return_value = FakeResponse(200, {'status': 'failed', 'error': 'MPI_ABORT'})
                with self.assertRaises(RuntimeError) as context:
                    worker(payload())
        self.assertIn('MPI_ABORT', str(context.exception))

    def test_max_wait_raises_with_last_status(self):
        worker = submitter(baseURL='http://server/api/jobs', poll=0, maxWait=0.5)
        worker.currentSubmission = 'runs/01'
        with patch('analysisManager.submitter.time.monotonic', side_effect=[0.0, 1.0]):
            with self.assertRaises(RuntimeError) as context:
                worker._wait()
        self.assertIn('Timed out', str(context.exception))
        self.assertIsNone(worker.currentSubmission)


if __name__ == '__main__':
    unittest.main()
