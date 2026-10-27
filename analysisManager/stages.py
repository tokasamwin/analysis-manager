from collections.abc import Callable
from typing import Any, Literal

def _basicGate(row) -> bool:
	return True

class stage(object):
	'''
	Object to wrap stages managed in a runManager
	args:
		function: function to run when called by runManager
	kwargs:
		args: the name of the args to use when calling the function. These must match args in the runManager.df DataFrame
		kwargs: the name of the kwargs to use when calling the function. These must match kwargs in the runManager.df DataFrame
		gatePair: A gate-check for this stage to proceed. a dictionary containing the names of values within the runManager.df DataFrame, and a boolean function to apply to the values. True-pass, False-fail
		name: Name of the stage. This is used when creating folders for the analysis stages
		outFile: the file to search for in the folder used for running this stage. The files matching the regex search of this string will be passed to the wrapped function, and can be used to avoid re-running analysis 
	'''
	@property
	def name(self):
		return self._name

	@name.setter
	def name(self,value) -> None:
		self._name=value

	def __init__(self,function: Callable,inputs: list[str] | None=None,gate: Callable | None=None,name: str | None=None,outFile: str='statepoint*h5') -> None:
		'''
		Stages: progressive steps of a design point assessment.
		inputs: a list of the names of the inputs to use when calling the function. These must match column names in the runManager.df DataFrame
		gate: a function that takes the row of the design point and returns True or False. If False, the stage will not be run, and the runManager will return a 'pass' value of '' (empty string) to indicate that the design point did not pass all stages.
			A stage may not progress if it does not pass a gate check. Gate checks assess a design point's suitability for further analysis. If a stage does not pass its gate check, the runManager will not progress to the next stage, and will return a 'pass' value of '' (empty string) to indicate that the design point did not pass all stages.
		name: the name of the stage. This is used when creating folders for the analysis stages
		outFile: the file to search for in the folder used for running this stage. The files matching the regex search of this string will be passed to the wrapped function, and can be used to avoid re-running analysis
		'''
		if name:
			self._name: str=name
		else:
			self._name: str=function.__name__
		if gate:
			self.gate: Callable[..., Any]=gate
			self.gateCheck = True
		else:
			self.gate: Callable[..., Literal[True]]=_basicGate #returns true with all inputs
			self.gateCheck = False
		if inputs:
			self.inputs: list[str]=inputs
		else:
			self.inputs=[]
		self.function: Callable[..., Any]=function
		self.outFile: str=outFile

	def __repr__(self) -> str:
		representation = 'Stage {}'
		if self.gateCheck: representation+=' with gate'
		return representation