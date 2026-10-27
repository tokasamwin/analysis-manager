"""Resolve callable references into real Python callables.

Two reference forms are supported (see ``resolveReference``):

* ``{"name": "package.module.function"}`` — a dotted import path, or a key in
  the registry (a JSON file of ``{"alias": "dotted.path"}`` mappings loaded
  from the ``ANALYSIS_STAGE_REGISTRY`` environment variable). The function is
  imported normally, so it works with ``ProcessPoolExecutor``-based runs.
* ``{"code": "lambda row: ...", "function": "optional_name"}`` — inline
  Python source, executed in a restricted namespace. Inline functions only
  work in-process (sequential or thread-based runs), because they cannot be
  pickled into fresh worker processes.
"""

import ast
import builtins
import importlib
import json
import os
import threading
from typing import Any, Callable

REGISTRY_ENV = 'ANALYSIS_STAGE_REGISTRY'

_lock = threading.Lock()
_registry: dict[str, str] = {}
_registryPath: str | None = None

# builtin names that inline code may use; everything else in :mod:`builtins`
# (open, eval, exec, __import__, getattr, ...) is deliberately omitted.
_SAFE_BUILTIN_NAMES = [
	'abs', 'all', 'any', 'ArithmeticError', 'AssertionError',
	'AttributeError', 'bool', 'BytesWarning', 'callable', 'complex',
	'dict', 'divmod', 'enumerate', 'Exception', 'filter', 'float',
	'format', 'frozenset', 'GeneratorExit', 'IOError', 'ImportError',
	'ImportWarning', 'IndexError', 'int', 'isinstance', 'issubclass',
	'KeyError', 'len', 'list', 'map', 'max', 'min', 'NameError',
	'next', 'NotImplementedError', 'OverflowError', 'print',
	'RuntimeError', 'set', 'slice', 'sorted', 'StopIteration',
	'SyntaxError', 'TypeError', 'tuple', 'ValueError',
	'ZeroDivisionError',
]

_FORBIDDEN_NAMES = {
	'os', 'sys', 'subprocess', 'socket', 'shutil', 'signal',
	'threading', 'multiprocessing', 'importlib', 'builtins',
	'__import__', 'exec', 'eval', 'compile', 'open', 'input',
	'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr',
	'breakpoint', 'exit', 'quit', 'memoryview', 'type',
}

class StageFunctionError(ValueError):
	"""Raised when a callable reference cannot be resolved safely."""

def _safeGlobals() -> dict[str, Any]:
	import math
	import numpy as _np
	import pandas as _pd
	return {
		'__builtins__': {
			name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES
		},
		# needed so that ``def`` statements work inside the restricted namespace
		'__build_class__': __build_class__,
		'np': _np,
		'pd': _pd,
		'math': math,
		'inf': float('inf'),
		'None': None,
		'True': True,
		'False': False,
	}

def _validateCode(code: str) -> None:
	tree = ast.parse(code)
	for node in ast.walk(tree):
		if isinstance(node, (ast.Import, ast.ImportFrom)):
			raise StageFunctionError(
				'Inline code may not import modules; register the function '
				'in the stage registry or reference an installed module'
			)
		if isinstance(node, ast.Name):
			if node.id.startswith('__') or node.id in _FORBIDDEN_NAMES:
				raise StageFunctionError(
					'Inline code may not use the name "{}"'.format(node.id)
				)

def codeToCallable(
	code: str,
	functionName: str | None = None,
) -> Callable[..., Any]:
	"""Compile inline code source into a callable.

	The source is either a single expression (for example
	``"lambda row: -row.get('volume', inf)"``), or a block that assigns /
	defines the callable under ``functionName``.
	"""
	if not(code or '').strip():
		raise StageFunctionError('Inline code is empty')
	code = code.strip()
	_validateCode(code)
	globalsDict = _safeGlobals()
	# First treat the source as a single expression (works for lambdas).
	try:
		value = eval(code, globalsDict)  # noqa: S307 - restricted namespace
	except SyntaxError:
		# Otherwise it defines the callable; look it up by name.
		slot = _SafeFunctionSlot()
		lookupName = functionName if functionName else 'function'
		globalsDict[lookupName] = slot
		exec(compile(code, '<inline code>', 'exec'), globalsDict)  # noqa: S102
		value = globalsDict.get(lookupName)
	if value is _SafeFunctionSlot() or value is None:
		raise StageFunctionError(
			'Inline code did not define a callable'
			+ (' named "{}"'.format(functionName) if functionName else '')
		)
	if not callable(value):
		raise StageFunctionError(
			'Inline code must evaluate to a callable'
			+ (' (expected "{}")'.format(functionName) if functionName else '')
		)
	return value

class _SafeFunctionSlot(object):
	"""Sentinel marking a slot that inline code must fill."""
	pass

def loadRegistry(path: str | None = None) -> dict[str, str]:
	"""Load the stage-function registry from a JSON file.

	``path`` defaults to the ``ANALYSIS_STAGE_REGISTRY`` environment variable.
	The file maps alias names to dotted import paths, for example::

		{"findArea": "mytools.geometry.findArea"}
	"""
	global _registry, _registryPath
	registryPath = str(path if path is not None
		else os.environ.get(REGISTRY_ENV, '')).strip()
	with _lock:
		if not(registryPath):
			_registry = {}
			_registryPath = None
			return {}
		try:
			with open(registryPath, 'r', encoding='utf-8') as f:
				mapping = json.load(f)
		except (OSError, json.JSONDecodeError) as error:
			raise StageFunctionError(
				'Registry file could not be read at {}: {}'.format(
					registryPath, error)
			)
		if not(isinstance(mapping, dict)
				and all(isinstance(key, str) and isinstance(value, str)
					for key, value in mapping.items())):
			raise StageFunctionError(
				'Registry file must be a JSON object mapping '
				'"alias": "dotted.path"'
			)
		_registry = mapping
		_registryPath = registryPath
		return dict(mapping)

def registry() -> dict[str, str]:
	"""Current registry mapping (empty if it has not been loaded)."""
	with _lock:
		return dict(_registry)

def _importDottedPath(dottedPath: str) -> Callable[..., Any]:
	try:
		moduleName, _, attributeName = dottedPath.rpartition('.')
		if not(moduleName):
			raise ImportError(dottedPath)
		module = importlib.import_module(moduleName)
		value: Any = module
		for part in attributeName.split('.'):
			value = getattr(value, part)
	except (ImportError, AttributeError) as error:
		raise StageFunctionError(
			'Could not import function "{}" ({})'.format(dottedPath, error)
		)
	if not callable(value):
		raise StageFunctionError(
			'Resolved reference "{}" is not callable'.format(dottedPath)
		)
	return value

def resolveReference(reference: dict | str) -> Callable[..., Any]:
	"""Resolve a callable reference from a run spec.

	Accepted forms:

	* ``"dotted.path"`` or ``{"name": "dotted.path"}``
	* ``{"name": "alias"}`` — looked up in the registry first, then imported
	  directly as a dotted path
	* ``{"code": "lambda row: ...", "function": "optional_name"}``
	"""
	if isinstance(reference, str):
		return _importDottedPath(reference.strip())
	if not(isinstance(reference, dict) and reference):
		raise StageFunctionError(
			'Callable references must be a dotted-path string, '
			'{"name": ...} or {"code": ...}'
		)
	name = reference.get('name')
	code = reference.get('code')
	if bool(name) == bool(code):
		raise StageFunctionError(
			'Callable references require exactly one of "name" or "code"'
		)
	if name is not None:
		if not isinstance(name, str):
			raise StageFunctionError('"name" must be a string')
		with _lock:
			currentRegistry = dict(_registry)
		if name in currentRegistry:
			return _importDottedPath(currentRegistry[name])
		return _importDottedPath(name)
	return codeToCallable(code, reference.get('function'))
