"""Minimal client for the analysisManager web API.

Usage::

	python examples/api_client_example.py            # default http://localhost:8080
	BASE=http://host:port python examples/api_client_example.py

Only the standard library is required.
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = 'http://localhost:8080'

SPEC = {
	"name": "example-api-run",
	"workers": "threads",
	# Stage functions here are inline code. They run in-process, which is
	# fine for the "threads" worker mode, and ideal when a stage simply
	# wraps an external analysis tool or library call.
	"stages": [
		{
			"name": "area",
			"inputs": ["depth", "width"],
			"function": {
				"code": "def f(depth, width, **kwargs):\n"
						"\treturn {'area': depth * width}",
				"function": "f",
			},
		},
		{
			"name": "volume",
			"inputs": ["depth", "width", "height"],
			"function": {
				"code": "def f(depth, width, height, **kwargs):\n"
						"\treturn {'volume': depth * width * height}",
				"function": "f",
			},
			"gate": {"code": "lambda row: row.get('area', 0) > 5"},
		},
	],
	"routines": [
		{
			"type": "search",
			"sampler": {
				"kind": "basic",
				"variables": {"depth": [1, 4], "width": [1, 4], "height": [1, 3]},
			},
			"density": 3,
			"method": "grid",
		},
		{
			"type": "optimise",
			"sampler": {
				"kind": "basic",
				"variables": {"depth": [1, 10], "width": [1, 10], "height": [1, 3]},
			},
			"target": {"code": "lambda row: -row.get('volume', 1e6)"},
			"pop_size": 8,
			"n_gen": 5,
		},
	],
}


def request(method: str, path: str, body=None, timeout: int = 120) -> tuple[int, bytes]:
	full = '{}{}'.format(BASE, path)
	data = None
	headers = {}
	if body is not None:
		data = json.dumps(body).encode('utf-8')
		headers['Content-Type'] = 'application/json'
	req = urllib.request.Request(full, data=data, headers=headers, method=method)
	try:
		with urllib.request.urlopen(req, timeout=timeout) as resp:
			return resp.status, resp.read()
	except urllib.error.HTTPError as error:
		return error.code, error.read()


def main() -> None:
	status, payload = request('GET', '/health')
	print('health:', status, payload.decode('utf-8'))

	status, payload = request('POST', '/api/runs', SPEC)
	if status != 202:
		print('submit failed:', status, payload.decode('utf-8'))
		sys.exit(1)
	run_id = json.loads(payload)['run_id']
	print('submitted run:', run_id)

	while True:
		status, payload = request('GET', '/api/runs/{}'.format(run_id))
		state = json.loads(payload)
		if status != 200:
			print('status error:', status, payload.decode('utf-8'))
			sys.exit(1)
		progress = state.get('progress', {})
		print('  status={} | designPointsDone={} | routines={}'.format(
			state['status'],
			progress.get('completed_design_points'),
			[(r['name'], r['state']) for r in progress.get('routines', [])],
		))
		if state['status'] in ('succeeded', 'failed'):
			if state['status'] == 'failed':
				print('  error:', state.get('error'))
				sys.exit(1)
			break
		time.sleep(2)

	status, payload = request('GET', '/api/runs/{}/results?format=csv'.format(run_id))
	print('results CSV (first 400 chars):')
	print(payload.decode('utf-8')[:400])


if __name__ == '__main__':
	main()
