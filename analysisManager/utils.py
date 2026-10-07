from numpy import str_ as npString
from hashlib import shake_128
from struct import pack
from pandas import Series
from collections.abc import Iterable
iterableCheck = lambda i: issubclass(type(i),Iterable) and not isinstance(i,str)


def hashRow(
	row: Series | dict,
	name: str | None=None,
	**updates):
	'''
	Internal function. Used to generate IDs for a given row.
	Intended usage: invoked by a runManager to create an ID column in the runManager.df DataFrame. This creates unique IDs based on the name and values of inputs and the run configuration. The intention is to create unique IDs that are specific to the analysis run, so that duplicate datapoints can be avoided, and datapoints can be reused across multiple analyses.
	'''
	data = dict(row)
	data.update(updates)
	inputs = list(data.values())
	if not name: name=''
	array=bytearray(name.encode())
	iterables=True
	while iterables:
		newInputs=[]
		for i,val in enumerate(inputs):
			if iterableCheck(val):
				newInputs+=list(val)
			else:
				newInputs.append(val)
			inputs=newInputs
		iterables=any([iterableCheck(i) for i in inputs])
	for value in inputs:
		if type(value) in [str,npString]:
			array+=str(value).encode(encoding='utf-8')
		elif hasattr(value,'__float__'):
			obj=pack('<f',float(value))
			for byte in obj:
				array.append(byte)
	hashval=shake_128(array).hexdigest(6)
	return hashval

def getDpInputs(
	row: Series | dict,
	inputs: list[str] | None=None,
	default = None
	) -> list:
	if inputs:
		args = [row.get(inp,default) for inp in inputs]
		args = list(filter(lambda i: i!=default, args))
	else:
		args = [row[v] for v in row.keys()]
	return args