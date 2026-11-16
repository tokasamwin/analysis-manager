import requests
from typing import Any
import time
TERMINAL_STATES = {"complete", "failed"}
TRANSIENT_STATES = {"queued", "running"}

class submitter(object):
	'''
	Object to provide a submission endpoint. Stages may run directly, or can call workers to run the stages.
	Instances of this class are optional to stages, and are a way to provide a submission endpoint without analysisManager being integrated into the environment for these tools. This also allows analysisManager to manage non-python tools accessible using requests.
	'''
	def __init__(self,baseURL: str,timeout: float = 60.0, poll: float = 5, directoryMapping: dict[str, str] | str | None = None, maxWait: float | None = None, **payloadItems) -> None:
		'''
		baseURL: the base URL of the submission endpoint. This is used to submit stages to the endpoint, and to poll for results.
		timeout: the timeout for the submission endpoint. This is used to determine how long to wait for a response from the endpoint before timing out.
		poll: the polling interval for the submission endpoint. This is used to determine how often to poll for results from the endpoint.
		directoryMapping: a mapping of directories for the submission endpoint.
		maxWait: optional total time in seconds to wait for a submission to finish. If it is exceeded, a RuntimeError is raised with the last known job status. Defaults to None (wait indefinitely).
		'''
		self.baseURL = baseURL
		self.timeout = timeout
		self.poll = poll
		self.maxWait = maxWait
		self.currentSubmission = None
		self.directoryMapping = directoryMapping
		self.payloadItems = payloadItems

	def _mapDirectories(self, path: str) -> str:
		'''
		Map directories for the submission endpoint. This is used to map directories for the submission endpoint.
		'''
		
		if isinstance(self.directoryMapping, dict):
			for src, dst in self.directoryMapping.items():
				path = str(path).replace(src, dst)
		elif isinstance(self.directoryMapping, str):
			src, dst = self.directoryMapping.split(':')
			path = str(path).replace(src, dst)
		if path.startswith('/'):
			path = path[1:]
		return path

	def _fail(self, message: str) -> RuntimeError:
		'''
		Clear the current submission and build an error for a failed poll.
		'''
		self.currentSubmission = None
		return RuntimeError(message)

	def _submit(self, payload: dict[str, Any]) -> dict[str, Any]:
		'''
		Submit a payload to the submission endpoint. This is used to submit stages to the endpoint, and to poll for results.
		payload: the payload to submit to the endpoint. This is used to submit stages to the endpoint, and to poll for results.
		'''
		if self.directoryMapping:
			payload['run_path'] = self._mapDirectories(payload['run_path'])
		for item in self.payloadItems:
			if self.payloadItems[item] is not None:
				payload[item] = self.payloadItems[item]
		try:
			response = requests.post(self.baseURL, json=payload, timeout=self.timeout)
		except requests.exceptions.RequestException as e:
			raise RuntimeError(f"Failed to submit payload: {e}")
		if not response.ok:
			raise RuntimeError(f"Submission to {self.baseURL} failed with HTTP {response.status_code}: {response.text[:500]}")
		try:
			result = response.json()
		except ValueError as e:
			raise RuntimeError(f"Submission to {self.baseURL} returned invalid JSON (HTTP {response.status_code}): {response.text[:500]}") from e
		if not isinstance(result, dict) or 'status' not in result:
			raise RuntimeError(f"Submission to {self.baseURL} returned an unexpected response (HTTP {response.status_code}): {result!r}")
		status = result.get('status')
		if status not in TERMINAL_STATES and status not in TRANSIENT_STATES:
			raise RuntimeError(f"Submission to {self.baseURL} returned an unexpected job status {status!r}: {result!r}")
		self.currentSubmission = payload['run_path']
		return result

	def _wait(self) -> dict[str, Any]:
		'''
		Wait for the current submission to complete. This is used to poll for results from the endpoint.
		'''
		if not self.currentSubmission:
			raise RuntimeError("No current submission to wait for.")
		runPath = self.currentSubmission
		responseDir = f"{self.baseURL}/{runPath}"
		start = time.monotonic()
		lastStatus = None
		while True:
			if self.maxWait is not None and time.monotonic() - start > self.maxWait:
				raise self._fail(
					f"Timed out after {self.maxWait}s waiting for {runPath!r} from {self.baseURL}; "
					f"last status was {lastStatus!r}"
				)
			try:
				response = requests.get(responseDir, timeout=self.timeout)
			except requests.exceptions.RequestException as e:
				raise self._fail(
					f"Failed to poll for results from {responseDir}: {e}. "
					"The server may have crashed or restarted, or may be unreachable."
				)
			if response.status_code == 404:
				raise self._fail(
					f"Job {runPath!r} was accepted by {self.baseURL} but the server now returns HTTP 404. "
					"The server has probably crashed or restarted and lost track of the job. "
					"Check the server logs and restart count, e.g.:\n"
					"  docker logs -t <container> | tail -100\n"
					"  docker inspect -f '{{.RestartCount}}' <container>"
				)
			if response.status_code != 200:
				raise self._fail(
					f"Polling {responseDir} failed with HTTP {response.status_code}: {response.text[:500]}"
				)
			try:
				result = response.json()
			except ValueError as e:
				raise self._fail(
					f"Polling {responseDir} returned invalid JSON (HTTP {response.status_code}): {response.text[:500]}"
				) from e
			status = result.get('status') if isinstance(result, dict) else None
			if status in TERMINAL_STATES:
				self.currentSubmission = None
				return result
			if status not in TRANSIENT_STATES:
				raise self._fail(
					f"Polling {responseDir} returned an unexpected job status {status!r}: {result!r}"
				)
			lastStatus = status
			time.sleep(self.poll)

	def __call__(self, payload: dict[str, Any], **kwargs) -> dict[str, Any]:
		if 'outFile' in payload['kwargs']:
			outFile = payload['kwargs'].get('outFile')
			if outFile:
				payload['kwargs']['outFile'] = self._mapDirectories(outFile)
		submission = self._submit(payload)
		status = submission.get('status')
		if status == 'complete':
			self.currentSubmission = None
			return submission['result']

		if status == 'failed':
			self.currentSubmission = None
			raise RuntimeError(f"Submission failed: {submission}")

		if status not in TRANSIENT_STATES:
			raise RuntimeError(f"Unexpected submission response: {submission}")

		result = self._wait()
		if result.get('status') != 'complete':
			raise RuntimeError(f"Submission failed: {result}")
		return result['result']
