"""Durable, restart-safe records for server runs.

The run manager (``analysisManager.runManager``) only persists per-design-point
artifacts; the consolidated results DataFrame it accumulates in ``runManager.df``
lives in memory and is lost when the server process exits. This module closes
that gap for the web layer:

* it writes a per-run ``run.json`` status record and a joined ``results.csv``,
  both atomically, into the run's base folder;
* on server start it re-indexes whatever is on disk and re-marks runs that
  died mid-flight (``queued``/``running``) as ``interrupted``;
* it provides readers so the Flask layer can fall back to disk for runs that
  are no longer in live memory.

Everything here is a direct pandas / stdlib export -- no new dependencies are
introduced.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

RUN_JSON = 'run.json'
RESULTS_CSV = 'results.csv'

#: run states that are no longer in progress and therefore "finished"
_TERMINAL_STATES = {'succeeded', 'failed', 'interrupted'}
#: states that mean the run had not yet completed when the process died
_INPROGRESS_STATES = {'queued', 'running'}


@dataclass
class RunRecord:
	"""A serialisable snapshot of a run's durable status.

	This is intentionally independent of the live ``server.Run`` object so that
	``server`` can hydrate a record from the in-memory state and
	``durable`` can persist it without importing ``server`` (avoids a cycle).
	"""

	run_id: str
	name: str
	status: str
	basePath: str
	spec: Optional[dict] = None
	workers: Optional[dict] = None
	submitted_at: Optional[float] = None
	started_at: Optional[float] = None
	finished_at: Optional[float] = None
	error: Optional[str] = None
	traceback: Optional[str] = None
	version: Optional[str] = None
	routines: list[dict] = field(default_factory=list)

	def to_dict(self) -> dict[str, Any]:
		payload = {
			'run_id': self.run_id,
			'name': self.name,
			'status': self.status,
			'basePath': self.basePath,
			'spec': self.spec,
			'workers': self.workers,
			'submitted_at': self.submitted_at,
			'started_at': self.started_at,
			'finished_at': self.finished_at,
			'error': self.error,
			'traceback': self.traceback,
			'version': self.version,
			'routines': self.routines,
		}
		return payload

	@classmethod
	def from_dict(cls, data: dict) -> 'RunRecord':
		known = {f for f in cls.__dataclass_fields__}
		return cls(**{k: v for k, v in data.items() if k in known})


def atomicWrite(path: Path, data: bytes) -> None:
	"""Write ``data`` to ``path`` atomically (tmp file + ``os.replace``)."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp = path.with_name(path.name + '.tmp')
	with open(tmp, 'wb') as handle:
		handle.write(data)
		handle.flush()
		os.fsync(handle.fileno())
	os.replace(tmp, path)


def resultsToCSV(frame: pd.DataFrame) -> str:
	"""Serialise a results DataFrame to CSV (inf -> empty), pandas-native.

	Shared by the live results endpoint and the durable ``results.csv`` so
	both produce identical bytes.
	"""
	frame = frame.copy(deep=True)
	for column in frame.columns:
		if pd.api.types.is_numeric_dtype(frame[column]):
			frame[column] = frame[column].where(np.isfinite(frame[column]))
	return frame.to_csv()


def runJsonPath(basePath) -> Path:
	return Path(basePath) / RUN_JSON


def resultsCsvPath(basePath) -> Path:
	return Path(basePath) / RESULTS_CSV


def writeRunJson(basePath, record: RunRecord) -> Path:
	"""Atomically (re)write the run's ``run.json`` reflecting ``record``."""
	basePath = Path(basePath)
	record.basePath = str(basePath)
	payload = json.dumps(record.to_dict(), indent=2, ensure_ascii=False)
	path = runJsonPath(basePath)
	atomicWrite(path, payload.encode('utf-8'))
	return path


def finalizeRun(basePath, record: RunRecord, df: Optional[pd.DataFrame]) -> None:
	"""Persist terminal state for a finished run.

	Writes ``results.csv`` (best effort: only when there is a non-empty frame),
	then writes ``run.json`` last, so a reader that sees ``status == succeeded``
	can always assume ``results.csv`` is present.
	"""
	basePath = Path(basePath)
	if df is not None and not df.empty:
		atomicWrite(resultsCsvPath(basePath), resultsToCSV(df).encode('utf-8'))
	writeRunJson(basePath, record)


def markStatus(basePath, record: RunRecord) -> None:
	"""Write a progress record (queued / running transition)."""
	writeRunJson(basePath, record)


def _iterRunJson(dataDir):
	"""Yield ``(basePath, runRecord)`` for every run.json under dataDir."""
	dataDir = Path(dataDir)
	if not dataDir.is_dir():
		return
	for candidate in sorted(dataDir.glob('*/' + RUN_JSON)):
		try:
			with open(candidate, 'r', encoding='utf-8') as handle:
				payload = json.load(handle)
		except (OSError, json.JSONDecodeError):
			continue
		yield candidate.parent, RunRecord.from_dict(payload)


def reindexRuns(dataDir) -> dict[str, str]:
	"""Re-scan ``dataDir`` and re-mark runs that died mid-flight.

	Any run whose on-disk status is still ``queued`` or ``running`` is rewritten
	as ``interrupted``. Returns a ``{run_id: basePath}`` index.
	"""
	import time
	index: dict[str, str] = {}
	for basePath, record in _iterRunJson(dataDir):
		index[record.run_id] = str(basePath)
		if record.status in _INPROGRESS_STATES:
			record.status = 'interrupted'
			record.finished_at = record.finished_at or time.time()
			if record.error is None:
				record.error = ('the server process stopped while this run '
								'was in progress; per-design-point artifacts '
								'are preserved, but the joined results were not')
			writeRunJson(basePath, record)
	return index


def loadRunFromDisk(dataDir, key) -> Optional[RunRecord]:
	"""Return a run record from disk by name (folder) or run_id, else None."""
	dataDir = Path(dataDir)
	# fast path: the run name *is* the folder name
	direct = dataDir / str(key) / RUN_JSON
	if direct.is_file():
		try:
			with open(direct, 'r', encoding='utf-8') as handle:
				return RunRecord.from_dict(json.load(handle))
		except (OSError, json.JSONDecodeError):
			pass
	# slow path: match the run_id field
	for _basePath, record in _iterRunJson(dataDir):
		if record.run_id == str(key):
			record.basePath = str(_basePath)
			return record
	return None


def listRuns(dataDir, limit: Optional[int] = None) -> list[dict]:
	"""List on-disk runs, newest (by finished_at) first."""
	records = [record for _basePath, record in _iterRunJson(dataDir)]

	def sortKey(record: RunRecord) -> tuple:
		# finished last (None) sorts after any timestamp
		return (record.finished_at is not None, record.finished_at or 0.0)

	records.sort(key=sortKey, reverse=True)
	if limit is not None:
		records = records[:limit]
	return [record.to_dict() for record in records]


def readResultsCsv(basePath) -> Optional[str]:
	"""Return the durable results CSV body, or None if it was not written."""
	path = resultsCsvPath(basePath)
	if not path.is_file():
		return None
	return path.read_text(encoding='utf-8')
