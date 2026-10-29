"""Flask web API for analysisManager.

Accepts run specs (samplers, stages and routines) as JSON, executes them
in this process behind a FIFO queue of one run at a time, and exposes
status, results and a dashboard per run.

Run the app with::

	gunicorn analysisManager.server:app --bind 0.0.0.0:8080 --workers 1 --threads 8

or, for development::

	python -m analysisManager.server
"""

import importlib.metadata
import json
import os
import queue
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, Response, send_file

import analysisManager as am
from analysisManager import (
	basicSampler,
	definedSampler,
	linearDistribution,
	exponentialDistribution,
	setDistribution,
	stage as _stage,
	runManager,
	searchRoutine,
	optimiseRoutine,
	surrogateRoutine,
)
from analysisManager.sampling import _methods as _samplingMethods

from analysisManager.routineDashboard import routineDashboard
from analysisManager import durable
from analysisManager.stage_functions import (
	resolveReference,
	registry,
	StageFunctionError,
)

try:
	import matplotlib
	matplotlib.use('Agg')
except ImportError:
	matplotlib = None

WORKERS_ENV = 'ANALYSIS_WORKERS'
VALID_WORKER_MODES = {'none', 'threads', 'processes'}

DATA_DIR = Path(am.config['path']['dataDir'])

class SpecError(ValueError):
	"""Raised when a run spec is invalid; reported to the client as HTTP 422."""

def version() -> str:
	try:
		return importlib.metadata.version('analysis-manager')
	except importlib.metadata.PackageNotFoundError:
		return '0.1.0'

def _record(run) -> 'durable.RunRecord':
	'''Snapshot the live Run state into a durable run record.'''
	return durable.RunRecord(
		run_id=run.runID,
		name=run.name,
		status=run.status,
		basePath=run.meta['basePath'],
		spec=run.spec,
		workers={
			'mode': run.meta.get('workersMode'),
			'count': run.meta.get('workerCount'),
		},
		submitted_at=run.submittedAt,
		started_at=run.startedAt,
		finished_at=run.finishedAt,
		error=run.error,
		traceback=run.meta.get('traceback'),
		version=version(),
		routines=list(run.routineStates),
	)

def _diskRunPublic(record: 'durable.RunRecord') -> dict:
	'''Present a durable record in the same shape as ``Run.public()``.'''
	completed = 0
	basePath = Path(record.basePath)
	if basePath.is_dir():
		completed = len(list(basePath.glob('*/results.json')))
	return {
		'run_id': record.run_id,
		'status': record.status,
		'name': record.name,
		'basePath': record.basePath,
		'queue_position': None,
		'error': record.error,
		'submitted_at': record.submitted_at,
		'started_at': record.started_at,
		'finished_at': record.finished_at,
		'version': record.version,
		'spec': record.spec,
		'progress': {
			'completed_design_points': completed,
			'routines': record.routines,
		},
	}

# ---------------------------------------------------------------------------
# spec -> objects
# ---------------------------------------------------------------------------

def _requireString(value, where) -> str:
	if not isinstance(value, str) or not value.strip():
		raise SpecError('{} must be a non-empty string'.format(where))
	return value

def _toFloatList(value, where, length=2) -> tuple[float, float]:
	if not(isinstance(value, (list, tuple)) and len(value) == length):
		raise SpecError(
			'{} must be a list of {} numbers'.format(where, length))
	try:
		return tuple(float(item) for item in value)
	except (TypeError, ValueError) as error:
		raise SpecError('{} must contain numbers: {}'.format(where, error))

def buildSampler(spec) -> object:
	'''Build a sampler from its spec.

	* ``{"kind": "basic", "variables": {"name": [low, high], ...}}``
	* ``{"kind": "defined", "variables": {"name":
	  {"type": "set", "vals": [...]}} | {"type": "linear", "limits": [lo, hi], "density": n} |
	  {"type": "exponential", "limits": [lo, hi], "density": n, "base": 10}, ...}}``
	'''
	if not(isinstance(spec, dict) and spec.get('kind') and isinstance(spec.get('variables', {}), dict)):
		raise SpecError('Sampler specs require "kind" and a "variables" object')
	variables = spec['variables']
	kind = spec['kind']
	if kind == 'basic':
		return basicSampler(
			**{name: _toFloatList(value, 'variable {}'.format(name))
				for name, value in variables.items()})
	if kind == 'defined':
		built = {}
		for name, value in variables.items():
			if not(isinstance(value, dict) and value.get('type')):
				raise SpecError(
					'defined sampler variable {} must specify a "type"'.format(name))
			varType = value['type']
			if varType == 'set':
				vals = value.get('vals')
				if not(isinstance(vals, list) and vals):
					raise SpecError('variable {} "vals" must be a non-empty list'.format(name))
				built[name] = setDistribution(vals)
			elif varType == 'linear':
				limits = _toFloatList(value.get('limits'), 'variable {} limits'.format(name))
				density = int(value.get('density', 5))
				built[name] = linearDistribution(limits, density=density)
			elif varType == 'exponential':
				limits = _toFloatList(value.get('limits'), 'variable {} limits'.format(name))
				density = int(value.get('density', 5))
				base = float(value.get('base', 10))
				built[name] = exponentialDistribution(limits, density=density, base=base)
			else:
				raise SpecError(
					'variable {} has unknown type "{}" (expected set, linear, or exponential)'
					.format(name, varType))
		return definedSampler(**built)
	raise SpecError('Unknown sampler kind "{}" (expected basic or defined)'.format(kind))

def buildStages(stageSpecs) -> tuple[list[_stage], bool]:
	'''Build stage objects from specs. Returns (stages, usesInlineCode).'''
	if not(isinstance(stageSpecs, list) and stageSpecs):
		raise SpecError('"stages" must be a non-empty list')
	stages = []
	usesInline = False
	for n, stageSpec in enumerate(stageSpecs):
		label = 'stages[{}]'.format(n)
		if not(isinstance(stageSpec, dict) and stageSpec):
			raise SpecError('{} must be an object'.format(label))
		functionRef = stageSpec.get('function')
		if functionRef is None:
			raise SpecError('{} must provide a "function" reference'.format(label))
		function = resolveReference(functionRef)
		if isinstance(functionRef, dict) and isinstance(functionRef.get('code'), str):
			usesInline = True
		gate = None
		if stageSpec.get('gate') is not None:
			gate = resolveReference(stageSpec['gate'])
			if isinstance(stageSpec['gate'], dict) and isinstance(stageSpec['gate'].get('code'), str):
				usesInline = True
		inputs = stageSpec.get('inputs')
		if inputs is not None and not(isinstance(inputs, list)
				and all(isinstance(item, str) for item in inputs)):
			raise SpecError('{} "inputs" must be a list of column names'.format(label))
		outFile = stageSpec.get('outFile', 'statepoint*h5')
		stages.append(_stage(
			function,
			inputs=inputs,
			gate=gate,
			name=stageSpec.get('name'),
			outFile=outFile,
		))
	return stages, usesInline

_OPTIMISE_OPTION_KEYS = ('pop_size', 'n_offsprings', 'n_gen', 'seed')

def buildRoutines(routineSpecs) -> tuple[list, bool]:
	'''Build routine objects from specs. Returns (routines, usesInlineCode).'''
	if not(isinstance(routineSpecs, list) and routineSpecs):
		raise SpecError('"routines" must be a non-empty list')
	routines = []
	usesInline = False
	for n, routineSpec in enumerate(routineSpecs):
		label = 'routines[{}]'.format(n)
		if not(isinstance(routineSpec, dict) and routineSpec.get('type')
				and routineSpec.get('sampler')):
			raise SpecError(
				'{} must provide a "type" and a "sampler"'.format(label))
		routineType = routineSpec['type']
		sampler = buildSampler(routineSpec['sampler'])
		inlineHere = False
		if routineType == 'search':
			kwargs = {}
			if 'density' in routineSpec:
				kwargs['density'] = routineSpec['density']
			if 'method' in routineSpec:
				if routineSpec['method'] not in _samplingMethods:
					errLabel = '{} "method" must be in:\n\t'.format(routineSpec['method']) + '\n\t'.join(_samplingMethods)
					raise SpecError(errLabel)
				kwargs['method'] = routineSpec['method']
			routines.append(searchRoutine(sampler, **kwargs))
		elif routineType == 'optimise':
			targetRef = routineSpec.get('target')
			if targetRef is None:
				raise SpecError(
					'{} must provide a "target" function reference'.format(label))
			targetFunction = resolveReference(targetRef)
			if isinstance(targetRef, dict) and isinstance(targetRef.get('code'), str):
				inlineHere = True
			kwargs = {}
			for key in _OPTIMISE_OPTION_KEYS:
				if key in routineSpec:
					kwargs[key] = routineSpec[key]
			routines.append(optimiseRoutine(sampler, targetFunction=targetFunction, **kwargs))
		elif routineType == 'surrogate':
			targets = routineSpec.get('targets')
			if targets is None:
				raise SpecError(
					'{} must provide "targets" (result column names to model)'.format(label))
			if isinstance(targets, str):
				targets = [targets]
			if not(isinstance(targets, list) and targets):
				raise SpecError(
					'{} "targets" must be a non-empty list of column names'.format(label))
			kwargs = {}
			for key in ('startDensity', 'aTol', 'rTol', 'method'):
				if key in routineSpec:
					kwargs[key] = routineSpec[key]
			if 'model' in routineSpec:
				model = resolveReference(routineSpec['model'])
				if isinstance(routineSpec['model'], dict) and isinstance(routineSpec['model'].get('code'), str):
					inlineHere = True
				kwargs['model'] = model
			routines.append(surrogateRoutine(sampler, targetOutputs=targets, **kwargs))
		else:
			raise SpecError(
				'{} has unknown type "{}" (expected search, optimise, or surrogate)'
				.format(label, routineType))
		usesInline = usesInline or inlineHere
	return routines, usesInline

def specToManager(spec) -> tuple[runManager, dict]:
	'''Validate a run spec and build the runManager for it.

	Returns ``(manager, meta)`` where ``meta`` carries run metadata such as
	whether inline code is present.
	'''
	if not(isinstance(spec, dict) and spec):
		raise SpecError('Run spec must be a JSON object')
	name = str(spec.get('name') or 'run-{}'.format(uuid.uuid4().hex[:8]))
	stages, stagesInline = buildStages(spec.get('stages'))
	routines, routinesInline = buildRoutines(spec.get('routines'))
	usesInline = stagesInline or routinesInline
	configuration = spec.get('configuration')
	if configuration is not None and not(isinstance(configuration, dict) and configuration):
		raise SpecError('"configuration" must be a JSON object when provided')
	manager = runManager(
		stages=stages,
		routines=routines,
		name=name,
		configuration=configuration,
		verbose=False,
	)
	mode = spec.get('workers', 'threads')
	if not isinstance(mode, str) or mode not in VALID_WORKER_MODES:
		raise SpecError(
			'"workers" must be one of: none, threads, processes')
	if mode == 'processes' and usesInline:
		raise SpecError(
			'workers="processes" is incompatible with inline "code" functions, '
			'because inline functions cannot be pickled into worker processes. '
			'Use workers="threads" or move the functions into a registered module.'
		)
	countValue = spec.get('workerCount', int(os.environ.get(WORKERS_ENV, 4)))
	try:
		count = int(countValue)
	except (TypeError, ValueError):
		raise SpecError('"workerCount" must be an integer')
	if count < 1 or count > 256:
		raise SpecError('"workerCount" must be between 1 and 256')
	meta = {
		'name': name,
		'usesInline': usesInline,
		'workersMode': mode,
		'workerCount': count,
		'basePath': str(manager.basePath),
	}
	manager._analysisMeta = meta   # attached for reference during execution
	return manager, meta

# ---------------------------------------------------------------------------
# run state + queue
# ---------------------------------------------------------------------------

class Run(object):
	'''State for a single submitted run.'''

	def __init__(self, runID: str, manager: runManager, meta: dict, spec: dict | None = None):
		self.runID = runID
		self.manager = manager
		self.meta = dict(meta)
		self.spec = spec
		self.status = 'queued'
		self.queuePosition = 0
		self.error = None
		self.submittedAt = time.time()
		self.startedAt = None
		self.finishedAt = None
		self.routineStates: list[dict] = [
			{'name': routine.name, 'state': 'pending'} for routine in manager.routines
		]

	def public(self) -> dict:
		return {
			'run_id': self.runID,
			'status': self.status,
			'name': self.name,
			'basePath': self.meta['basePath'],
			'queue_position': self.queuePosition,
			'error': self.error,
			'submitted_at': self.submittedAt,
			'started_at': self.startedAt,
			'finished_at': self.finishedAt,
			'progress': {
				'completed_design_points': self._completedDPs(),
				'routines': self.routineStates,
			},
		}

	@property
	def name(self) -> str:
		return self.meta['name']

	def _completedDPs(self) -> int:
		basePath = Path(self.meta['basePath'])
		if not basePath.is_dir():
			return 0
		return len(list(basePath.glob('*/results.json')))

def _resolveWorkers(run: Run, routine: object):
	'''Compute the workers argument for one routine of one run.'''
	mode = run.meta['workersMode']
	count = run.meta['workerCount']
	if mode == 'none':
		return 1
	if mode == 'processes':
		return count
	# threads mode: keep everything inside this process.
	# pymoo needs a plain int and spawns its own processes, so optimises run
	# in-process; search/surrogate use a thread pool (which suits stage
	# functions that wrap external binaries).
	if isinstance(routine, optimiseRoutine):
		return 1
	return ThreadPoolExecutor(max_workers=max(count, 1))

# ---------------------------------------------------------------------------
# app setup
# ---------------------------------------------------------------------------

def createApp() -> Flask:
	'''Build the Flask app and start the run worker thread on import.

	At build time the durable run records on disk are re-indexed; any run that
	had not finished when the previous process exited is re-marked as
	``interrupted`` so its on-disk state is honest.
	'''
	app = Flask(__name__)
	app.config['JSON_SORT_KEYS'] = False

	_runs: dict[str, Run] = {}
	_queue: queue.Queue[str] = queue.Queue()
	_lock = threading.Lock()
	durable.reindexRuns(DATA_DIR)

	@app.get('/health')
	def health() -> Response:
		return jsonify({
			'status': 'ok',
			'version': version(),
			'queued': _queue.qsize(),
			'runs': {
				key: value.status for key, value in list(_runs.items())
			},
		})

	@app.get('/api/config')
	def apiConfig() -> Response:
		return jsonify({
			'version': version(),
			'dataDir': str(DATA_DIR),
			'registry': registry(),
			'defaultWorkerMode': 'threads',
			'defaultWorkerCount': os.environ.get(WORKERS_ENV, 4),
		})

	@app.post('/api/runs')
	def submitRun() -> Response:
		spec = request.get_json(force=True, silent=True)
		if spec is None:
			return jsonify({'error': 'Request body must be a JSON object'}), 422
		try:
			manager, meta = specToManager(spec)
		except (SpecError, StageFunctionError, ValueError, TypeError) as error:
			return jsonify({'error': str(error)}), 422
		runID = uuid.uuid4().hex[:12]
		run = Run(runID, manager, meta, spec)
		run.queuePosition = _queue.qsize()
		with _lock:
			_runs[runID] = run
		durable.markStatus(Path(meta['basePath']), _record(run))
		_queue.put(runID)
		return jsonify({
			'run_id': runID,
			'status': 'queued',
			'name': run.name,
			'positions': {
				'queue': run.queuePosition,
				'get': '/api/runs/{}'.format(runID),
				'results': '/api/runs/{}/results?format=csv'.format(runID),
				'dashboard': '/api/runs/{}/dashboard'.format(runID),
			},
		}), 202

	@app.get('/api/runs')
	def listAllRuns() -> Response:
		'''List runs, live first (running state wins) then durable ones from disk.'''
		try:
			limit = int(request.args.get('limit'))
		except (TypeError, ValueError):
			limit = None
		byName = {record['name']: record for record in durable.listRuns(DATA_DIR)}
		live = list(_runs.values())
		for run in live:
			byName.pop(run.name, None)
			byName[run.name] = run.public()
		ordered = sorted(byName.values(),
			key=lambda record: (record.get('finished_at') is not None,
				record.get('finished_at') or 0.0),
			reverse=True)
		if limit is not None:
			ordered = ordered[:limit]
		return jsonify({
			'dataDir': str(DATA_DIR),
			'runs': ordered,
		})

	@app.get('/api/runs/<runID>')
	def getRun(runID: str) -> Response:
		with _lock:
			run = _runs.get(runID)
		if run is not None:
			return jsonify(run.public())
		# not in live memory: fall back to the durable record on disk
		record = durable.loadRunFromDisk(DATA_DIR, runID)
		if record is None:
			return jsonify({'error': 'Unknown run_id: {}'.format(runID)}), 404
		return jsonify(_diskRunPublic(record))

	@app.get('/api/runs/<runID>/results')
	def getResults(runID: str) -> Response:
		with _lock:
			run = _runs.get(runID)
		if run is not None:
			if run.status in ('queued', 'running'):
				return jsonify({'status': run.status, 'error': 'run is not finished yet'}), 409
			if run.status == 'failed':
				return jsonify({'status': run.status, 'error': run.error,
					'traceback': run.meta.get('traceback')}), 500
			frame = run.manager.df
			if (request.args.get('format') or 'csv').lower() == 'json':
				records = json.loads(frame.to_json(orient='records', force_ascii=False))
				body = json.dumps({'name': run.name, 'run_id': runID,
					'status': run.status, 'results': records}, indent=2)
				return Response(
					body, mimetype='application/json',
					headers={'Content-Disposition':
						'attachment; filename="{}-results.json"'.format(run.name)})
			return Response(
				durable.resultsToCSV(frame),
				mimetype='text/csv; charset=utf-8',
				headers={'Content-Disposition':
					'attachment; filename="{}-results.csv"'.format(run.name)})
		# not in live memory: serve the durable record from disk
		record = durable.loadRunFromDisk(DATA_DIR, runID)
		if record is None:
			return jsonify({'error': 'Unknown run_id: {}'.format(runID)}), 404
		if record.status == 'failed':
			return jsonify({'status': 'failed', 'error': record.error,
				'traceback': record.traceback}), 500
		body = durable.readResultsCsv(record.basePath)
		if body is None:
			return jsonify({
				'status': record.status,
				'error': 'no results file is stored for this run',
			}), 404
		return Response(
			body, mimetype='text/csv; charset=utf-8',
			headers={'Content-Disposition':
				'attachment; filename="{}-results.csv"'.format(record.name)})

	@app.get('/api/runs/<runID>/dashboard')
	def getDashboard(runID: str) -> Response:
		with _lock:
			run = _runs.get(runID)
		if run is None:
			return jsonify({'error': 'Unknown run_id: {}'.format(runID)}), 404
		frame = run.manager.df
		columns = [
			column for column in frame.columns
			if pd.api.types.is_numeric_dtype(frame[column])
		]
		outputPath = Path(run.meta['basePath']) / 'dashboard.html'
		dashboard = routineDashboard(run.manager, columns, output=outputPath)
		try:
			dashboard.refresh()
		except Exception as error:
			return jsonify({'error': 'Dashboard error: {}'.format(error)}), 500
		return send_file(str(outputPath), mimetype='text/html', download_name='dashboard.html')

	@app.delete('/api/runs/<runID>')
	def cancelRun(runID: str) -> Response:
		return jsonify({
			'error': 'Cancel not implemented yet (planned for v2); '
				'the run will complete or fail on its own'
		}), 501

	# --- worker thread (FIFO, one run at a time) -------------------------

	def _runRoutine(run: Run, routine: object) -> None:
		workers = _resolveWorkers(run, routine)
		try:
			run.manager.runRoutine(routine, workers=workers)
		finally:
			if isinstance(workers, ThreadPoolExecutor):
				workers.shutdown(wait=True)

	def _worker() -> None:
		while True:
			runID = _queue.get()
			with _lock:
				run = _runs.get(runID)
			if run is None:
				continue
			runBase = Path(run.meta['basePath'])
			run.status = 'running'
			run.startedAt = time.time()
			durable.markStatus(runBase, _record(run))
			try:
				for routine, state in zip(run.manager.routines, run.routineStates):
					state['state'] = 'running'
					try:
						_runRoutine(run, routine)
					finally:
						if state['state'] != 'error':
							state['state'] = 'done'
					durable.markStatus(runBase, _record(run))
				run.status = 'succeeded'
			except Exception as error:
				detail = traceback.format_exc()
				stateError = [
					state for state in run.routineStates
					if state['state'] == 'running'
				]
				if stateError:
					stateError[0]['state'] = 'error'
				run.status = 'failed'
				run.error = '{}: {}'.format(type(error).__name__, error)
				run.meta['traceback'] = detail
			finally:
				run.finishedAt = time.time()
				frame = getattr(run.manager, 'df', None)
				if frame is not None and frame.empty:
					frame = None
				try:
					durable.finalizeRun(runBase, _record(run), frame)
				except Exception:
					run.status = 'failed'
					run.error = (
						(run.error + ' | ' if run.error else '') +
						'failed to write durable run record: ' +
						traceback.format_exc_only().strip()
					)
					try:
						durable.writeRunJson(runBase, _record(run))
					except Exception:
						pass

	workerThread = threading.Thread(target=_worker, name='analysis-run-worker', daemon=True)
	workerThread.start()

	app._analysisRuns = _runs    # exposed for tests / introspection
	app._analysisQueue = _queue
	return app

app = createApp()

def main() -> None:
	'''Development server entry point: python -m analysisManager.server'''
	host = os.environ.get('ANALYSIS_HOST', '0.0.0.0')
	port = int(os.environ.get('ANALYSIS_PORT', 8080))
	app.run(host=host, port=port, threaded=True)

if __name__ == '__main__':
	main()
