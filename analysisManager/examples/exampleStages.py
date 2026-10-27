"""Example stage functions usable through the stage-function registry.

``docker/registry.example.json`` maps short aliases to these dotted paths so
that web-API run specs can reference them by alias, for example::

	{"name": "area", "inputs": ["depth", "width"],
	 "function": {"name": "findArea"}}
"""

def findArea(depth, width, **kwargs):
	"""Compute the area of a rectangle (toy stage for demos and tests)."""
	return {'area': depth * width}

def findVolume(depth, width, height, **kwargs):
	"""Compute the volume of a box (toy stage for demos and tests)."""
	return {'volume': depth * width * height}

def areaGate(row):
	"""Gate: only progress design points whose area is greater than five."""
	return bool(row.get('area', 0) > 5)
