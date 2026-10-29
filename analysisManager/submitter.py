import requests
from typing import Any
import time
TERMINAL_STATES = {"complete", "failed"}

class submitter(object):
	'''
	Object to provide a submission endpoint. Stages may run directly, or can call workers to run the stages.
	Instances of this class are optional to stages, and are a way to provide a submission endpoint without analysisManager being integrated into the environment for these tools. This also allows analysisManager to manage non-python tools accessible using requests.
	'''
	def __init__(self,baseURL: str,timeout: float = 60.0, poll: float = 5, directoryMapping: dict[str, str] | str | None = None,**payloadItems) -> None:
		'''
		baseURL: the base URL of the submission endpoint. This is used to submit stages to the endpoint, and to poll for results.
		timeout: the timeout for the submission endpoint. This is used to determine how long to wait for a response from the endpoint before timing out.
		poll: the polling interval for the submission endpoint. This is used to determine how often to poll for results from the endpoint.
		directoryMapping: a mapping of directories for the submission endpoint.
		'''
		self.baseURL = baseURL
		self.timeout = timeout
		self.poll = poll
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
			self.currentSubmission = payload['run_path']
			return response.json()
		except requests.exceptions.RequestException as e:
			raise RuntimeError(f"Failed to submit payload: {e}")

	def _wait(self) -> dict[str, Any]:
		'''
		Wait for the current submission to complete. This is used to poll for results from the endpoint.
		'''
		if not self.currentSubmission:
			raise RuntimeError("No current submission to wait for.")
		responseDir = f"{self.baseURL}/{self.currentSubmission}"
		while True:
			try:
				response = requests.get(responseDir, timeout=self.timeout)
				result = response.json()
			except requests.exceptions.RequestException as e:
				raise RuntimeError(f"Failed to poll for results: {e}")
			if result.get('status') in TERMINAL_STATES:
				self.currentSubmission = None
				return result
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

		if status != 'queued':
			raise RuntimeError(f"Unexpected submission response: {submission}")

		result = self._wait()
		if result.get('status') != 'complete':
			raise RuntimeError(f"Submission failed: {result}")
		return result['result']
