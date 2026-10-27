# -*- coding: utf-8 -*-
"""Tests for the analysisManager web server (analysisManager.server)."""

import json
import os
import sys
import tempfile
import time

import pytest

from analysisManager.server import app, specToManager, SpecError
from analysisManager import stage_functions
from analysisManager.stage_functions import (
	codeToCallable,
	loadRegistry,
	resolveReference,
	StageFunctionError,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def client():
	with app.test_client() as c:
		yield c

def _wait(client, run_id, timeout=180):
	deadline = time.time() + timeout
	state = None
	while time.time() < deadline:
		state = client.get('/api/runs/{}'.format(run_id)).get_json()
		if state['status'] in ('succeeded', 'failed'):
			return state
		time.sleep(0.5)
	return state

# Named top-level functions so the registry can point at them by dotted path.
# They must live in this module (importable by 'test_server') so that
# ProcessPoolExecutor child processes can unpickle them by reference.
def _areaFn(depth, width, **kwargs):
	return {'area': depth * width}

def _volumeFn(depth, width, height, **kwargs):
	return {'volume': depth * width * height}

def _gateAreaFn(row):
	return row.get('area', 0) > 5

def _pauseFn(a, **kwargs):
	time.sleep(0.4)
	return {'paused': True}

def _doubleFn():
	return 4

def _registryFile(tmp_path, mappings):
	"""Write a registry JSON file and load it into the stage_functions module."""
	regFile = tmp_path / 'registry.json'
	regFile.write_text(json.dumps(mappings))
	loadRegistry(str(regFile))
	return str(regFile)

def _inlineSpec():
	return {
		'name': 'test-inline-run-{}'.format(int(time.time() * 1000)),
		'workers': 'threads',
		'stages': [
			{
				'name': 'area',
				'inputs': ['depth', 'width'],
				'function': {
					'code': 'def f(depth, width, **kwargs):\n'
							'\treturn {"area": depth * width}',
					'function': 'f',
				},
			},
			{
				'name': 'volume',
				'inputs': ['depth', 'width', 'height'],
				'function': {
					'code': 'def f(depth, width, height, **kwargs):\n'
							'\treturn {"volume": depth * width * height}',
					'function': 'f',
				},
				'gate': {'code': 'lambda row: row.get("area", 0) > 5'},
			},
		],
		'routines': [
			{
				'type': 'search',
				'sampler': {
					'kind': 'basic',
					'variables': {'depth': [1, 4], 'width': [1, 4], 'height': [1, 3]},
				},
				'density': 2,
				'method': 'grid',
			},
			{
				'type': 'optimise',
				'sampler': {
					'kind': 'basic',
					'variables': {'depth': [1, 10], 'width': [1, 10], 'height': [1, 3]},
				},
				'target': {'code': 'lambda row: -row.get("volume", 1e6)'},
				'pop_size': 4,
				'n_gen': 2,
			},
		],
	}

# ---------------------------------------------------------------------------
# stage_functions unit tests
# ---------------------------------------------------------------------------

def test_code_expression_lambda():
	fn = codeToCallable('lambda row: -row.get("area", 0)')
	assert fn({'area': 4}) == -4
	assert fn({}) == -0

def test_code_uses_safe_globals():
	fn = codeToCallable('lambda row: max(0, row.get("a", 0)) + len("xy")')
	assert fn({'a': 1}) == 3

def test_code_named_function():
	fn = codeToCallable(
		'def f(depth, width, **kwargs):\n\treturn {"area": depth * width}',
		functionName='f',
	)
	assert fn(3, 4) == {'area': 12}

def test_code_without_function_name():
	with pytest.raises(StageFunctionError):
		codeToCallable('x = 1\nx')

def test_code_empty():
	with pytest.raises(StageFunctionError):
		codeToCallable('   ')

def test_code_forbidden_name():
	with pytest.raises(StageFunctionError):
		codeToCallable('lambda row: os.getpid()')

def test_code_forbidden_open():
	with pytest.raises(StageFunctionError):
		codeToCallable('lambda row: len(open("/etc/passwd").read())')

def test_code_import_blocked():
	with pytest.raises(StageFunctionError):
		codeToCallable('import subprocess')

def test_code_must_be_callable():
	with pytest.raises(StageFunctionError):
		codeToCallable('42')

def test_dotted_path_reference():
	fn = resolveReference('json.dumps')
	assert fn({'a': 1}) == '{"a": 1}'

def test_dotted_path_bad():
	with pytest.raises(StageFunctionError):
		resolveReference('no.such.module.anywhere.func')

def test_reference_requires_exactly_one():
	with pytest.raises(StageFunctionError):
		resolveReference({'name': 'json.dumps', 'code': '42'})
	with pytest.raises(StageFunctionError):
		resolveReference({})

def test_registry_loads_and_resolves(tmp_path):
	mapping = {'double': 'test_server._doubleFn'}
	regFile = _registryFile(tmp_path, mapping)
	assert loadRegistry(regFile) == mapping
	from analysisManager.stage_functions import registry
	assert registry() == mapping

def test_registry_bad_file(tmp_path):
	bad = tmp_path / 'bad.json'
	bad.write_text('not json')
	with pytest.raises(StageFunctionError):
		loadRegistry(str(bad))

# ---------------------------------------------------------------------------
# spec validation
# ---------------------------------------------------------------------------

def _baseSpec():
	return {
		'workers': 'threads',
		'stages': [
			{
				'name': 'area',
				'inputs': ['depth', 'width'],
				'function': {'code': 'lambda d, w: {"area": d * w}',
							 'function': 'f'},
			},
		],
		'routines': [
			{
				'type': 'search',
				'sampler': {'kind': 'basic', 'variables': {'a': [0, 1]}},
				'density': 2,
			},
		],
	}

def test_spec_not_object():
	with pytest.raises(SpecError):
		specToManager([])

def test_spec_missing_stages():
	spec = _baseSpec()
	del spec['stages']
	with pytest.raises(SpecError):
		specToManager(spec)

def test_spec_missing_routines():
	spec = _baseSpec()
	spec['routines'] = []
	with pytest.raises(SpecError):
		specToManager(spec)

def test_spec_unknown_sampler_kind():
	spec = _baseSpec()
	spec['routines'][0]['sampler']['kind'] = 'weird'
	with pytest.raises(SpecError):
		specToManager(spec)

def test_spec_unknown_routine_type():
	spec = _baseSpec()
	spec['routines'][0]['type'] = 'weird'
	with pytest.raises(SpecError):
		specToManager(spec)

def test_spec_optimise_needs_target():
	spec = _baseSpec()
	spec['routines'][0]['type'] = 'optimise'
	with pytest.raises(SpecError):
		specToManager(spec)

def test_spec_bad_workers():
	spec = _baseSpec()
	spec['workers'] = 'super'
	with pytest.raises(SpecError):
		specToManager(spec)

def test_spec_bad_worker_count():
	spec = _baseSpec()
	spec['workerCount'] = 'many'
	with pytest.raises(SpecError):
		specToManager(spec)

def test_spec_processes_with_inline_rejected():
	spec = _baseSpec()
	spec['workers'] = 'processes'
	with pytest.raises(SpecError):
		specToManager(spec)

# ---------------------------------------------------------------------------
# endpoint tests
# ---------------------------------------------------------------------------

def test_health(client):
	resp = client.get('/health')
	assert resp.status_code == 200
	body = resp.get_json()
	assert body['status'] == 'ok'
	assert 'version' in body

def test_config(client):
	resp = client.get('/api/config')
	assert resp.status_code == 200
	body = resp.get_json()
	assert 'dataDir' in body
	assert 'registry' in body

def test_unknown_run_404(client):
	assert client.get('/api/runs/nope').status_code == 404
	assert client.get('/api/runs/nope/results').status_code == 404
	assert client.get('/api/runs/nope/dashboard').status_code == 404

def test_cancel_not_implemented(client):
	assert client.delete('/api/runs/nope').status_code == 501

def test_invalid_spec_422(client):
	resp = client.post('/api/runs', json={'stages': []})
	assert resp.status_code == 422
	assert 'error' in resp.get_json()

def test_results_before_finish_409(client):
	run_id = client.post('/api/runs', json=_inlineSpec()).get_json()['run_id']
	resp = client.get('/api/runs/{}/results'.format(run_id))
	assert resp.status_code in (409, 200)
	_wait(client, run_id)
	# once finished, results must be available
	assert client.get('/api/runs/{}/results'.format(run_id)).status_code == 200

def test_end_to_end_inline(client):
	"""Full run with inline code in threads mode must finish and return data."""
	resp = client.post('/api/runs', json=_inlineSpec())
	assert resp.status_code == 202
	run_id = resp.get_json()['run_id']
	state = _wait(client, run_id)
	assert state['status'] == 'succeeded', state
	progress = state['progress']
	assert all(state['state'] == 'done'
				for state in progress['routines'])
	assert progress['completed_design_points'] >= 1

	results = client.get('/api/runs/{}/results?format=json'.format(run_id))
	assert results.status_code == 200
	body = results.get_json()
	rows = body['results']
	depths = [r['depth'] for r in rows if r.get('area') is not None]
	areas = [r['area'] for r in rows if r.get('area') is not None]
	widths = [r['width'] for r in rows if r.get('area') is not None]
	for d, a, w in zip(depths, areas, widths):
		assert float(a) == pytest.approx(float(d) * float(w))

dashboard = None

def test_end_to_end_registry_processes(client, tmp_path):
	"""Registry functions + workers=processes must finish and return data."""
	_registryFile(tmp_path, {
		'areaStage': 'test_server._areaFn',
		'volumeStage': 'test_server._volumeFn',
		'gateArea': 'test_server._gateAreaFn',
	})
	spec = {
		'name': 'test-registry-{}'.format(int(time.time() * 1000)),
		'workers': 'processes',
		'workerCount': 2,
		'stages': [
			{'name': 'area', 'inputs': ['depth', 'width'],
			 'function': {'name': 'areaStage'}},
			{'name': 'volume', 'inputs': ['depth', 'width', 'height'],
			 'function': {'name': 'volumeStage'}, 'gate': {'name': 'gateArea'}},
		],
		'routines': [
			{'type': 'search',
			 'sampler': {'kind': 'basic',
				 'variables': {'depth': [1, 4], 'width': [1, 4], 'height': [1, 3]}},
			 'density': 2, 'method': 'grid'},
		],
	}
	resp = client.post('/api/runs', json=spec)
	assert resp.status_code == 202
	run_id = resp.get_json()['run_id']
	state = _wait(client, run_id)
	assert state['status'] == 'succeeded', state
	results = client.get('/api/runs/{}/results?format=csv'.format(run_id))
	assert results.status_code == 200
	text = results.get_data(as_text=True)
	assert 'area' in text
	assert 'volume' in text

def test_fifo_queue_ordering(client, tmp_path):
	"""While one run executes, a second run must queue behind it."""
	_registryFile(tmp_path, {'pause': 'test_server._pauseFn'})
	def makeSpec(name):
		return {
			'name': name,
			'workers': 'threads',
			'stages': [{'name': 'pause', 'inputs': ['a'],
				'function': {'name': 'pause'}}],
			'routines': [
				{'type': 'search',
				 'sampler': {'kind': 'basic', 'variables': {'a': [0, 1]}},
				 'density': 2},
			],
		}
	rid1 = client.post('/api/runs', json=makeSpec('fifo-first-1')).get_json()['run_id']
	rid2 = client.post('/api/runs', json=makeSpec('fifo-first-2')).get_json()['run_id']
	state1 = _wait(client, rid1)
	state2 = _wait(client, rid2)
	assert state1['status'] == 'succeeded', state1
	assert state2['status'] == 'succeeded', state2
	# the second run must not start before the first one finished
	assert state2['started_at'] >= state1['finished_at'] - 0.5

def test_dashboard_available(client):
	spec = _inlineSpec()
	rid = client.post('/api/runs', json=spec).get_json()['run_id']
	state = _wait(client, rid)
	assert state['status'] == 'succeeded', state
	resp = client.get('/api/runs/{}/dashboard'.format(rid))
	assert resp.status_code == 200
	assert b'<!doctype html>' in resp.data.lower()

# ---------------------------------------------------------------------------
# _config environment behaviour (subprocess so the env is set before import)
# ---------------------------------------------------------------------------

def test_config_env_var_respected():
	import subprocess
	code = (
		'import analysisManager;\n'
		'd = analysisManager.config["path"]["dataDir"]\n'
		'assert d == __import__("os").environ["ANALYSISMANAGER_DATADIR"], d\n'
		'print("OK")'
	)
	tmp = tempfile.mkdtemp(prefix='amcfg_sub_')
	env = dict(os.environ, ANALYSISMANAGER_DATADIR=tmp)
	result = subprocess.run(
		[sys.executable, '-c', code],
		capture_output=True, text=True, timeout=60, env=env,
		cwd=os.path.join(os.path.dirname(os.path.dirname(__file__))),
	)
	assert result.returncode == 0, result.stderr

def test_config_non_tty_no_prompt():
	import subprocess
	"""A fresh $HOME with no settings.ini and a non-TTY stdin must not hang."""
	code = 'import analysisManager; print("OK")'
	freshHome = tempfile.mkdtemp(prefix='amhome_sub_')
	env = dict(os.environ)
	env.pop('ANALYSISMANAGER_DATADIR', None)
	env['HOME'] = freshHome
	env.pop('XDG_CONFIG_HOME', None)
	env.pop('XDG_DATA_HOME', None)
	result = subprocess.run(
		[sys.executable, '-c', code],
		stdin=subprocess.DEVNULL,
		capture_output=True, text=True, timeout=60, env=env,
		cwd=os.path.join(os.path.dirname(os.path.dirname(__file__))),
	)
	assert result.returncode == 0, result.stderr
