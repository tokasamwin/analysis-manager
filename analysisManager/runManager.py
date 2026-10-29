from analysisManager import config
from analysisManager.utils import getDpInputs, hashRow
from shutil import rmtree as removeFolder
from json import dump
from pathlib import Path
from copy import copy
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import pandas as pd

from analysisManager.stages import stage as _stage

dataDir: Path = Path(config['path']['dataDir'])

'''
This module contains tools to conduct analysis, including two separate tools for conducting analysis: a "searcher" class for conducting a search over pre-defined design points, and an "optimiser" class for optimising within a design space. Both of these are used to conduct analysis, which is defined by "stage" classes. Stages are used to create, run and manage multistaged analysis.

'''

class runManager(object):
	def __init__(
		self,
		stages: Sequence[_stage],
		routines: Sequence[Any],
		name: str = 'Default',
		configuration: Optional[dict[str, Any]] = None,
		verbose: bool = True,
	) -> None:
		self._verbose: bool = verbose
		self.stages: Sequence[_stage] = stages
		self.routines: Sequence[Any] = routines
		self.name: str = name
		self.df: pd.DataFrame = pd.DataFrame()
		if configuration:
			self.configuration: dict[str, Any] = configuration
		else:
			self.configuration = {}
		if configuration:
			self._configHash: str = hashRow(self.configuration)
		else:
			self._configHash = ''
		self.basePath: Path = dataDir / self.name
		if not(self.basePath.is_dir()): self.basePath.mkdir(parents=True)
		fname = self.basePath / 'config-{}.json'.format(self._configHash)
		with open(fname,'w') as f:
			dump(self.configuration,f,indent=4)
	
	def runDP(
		self,
		row: Union[pd.Series, dict[str, Any]],
		runText: Optional[str] = None,
	) -> dict[str, Any]:
		ID = row['ID']
		results = {}
		localPath = self.basePath / ID
		if not(localPath.is_dir()):
			localPath.mkdir()
		dictToDump = copy(dict(row))
		if self._configHash:
			dictToDump['configuration ID'] = self._configHash
		with open(localPath / 'inputs.json','w') as f:
			dump(dictToDump,f)
		for n,stage in enumerate(self.stages):
			args = getDpInputs(row,inputs = stage.inputs)
			nwd = self.basePath / ID / stage.name
			if not(nwd.is_dir()): nwd.mkdir(parents=True)
			files = list(nwd.glob(stage.outFile))
			if files:
				outFile = files[0]
			else:
				outFile = None
			if runText:
				localText = runText + ' | Stage {} of {} ({})'.format(n+1,len(self.stages),ID)
				if self._verbose: print(localText)
			if stage.gate(row):
				result = stage(*args,path=nwd,outFile=outFile,config = self.configuration)
				result['pass'] = result.get('pass',stage.name)
				row = row.to_dict()
				row.update(result)
				row = pd.Series(row)
				results.update(result)
			else:
				return {'pass':''}
		with open(localPath / 'results.json','w') as f:
			dump(results,f,indent=4)
		return results

	def runRoutine(
		self,
		routine: Any,
		workers: Any = None,
		**kwargs: Any,
	) -> None:
		localDF = routine(self.runDP,workers = workers,**kwargs).copy(deep=True)
		localDF['Routine'] = routine.name
		self.df = pd.concat([self.df,localDF],axis=0,ignore_index=True).reset_index(drop=True)

	def runRoutines(self, **kwargs: Any) -> None:
		for routine in self.routines:
			self.runRoutine(routine,**kwargs)
	
	def trimIDs(self, *IDs: str) -> None:
		for ID in IDs:
			folder = self.basePath / ID
			if folder.is_dir():
				removeFolder(folder)

if __name__ == '__main__':
	from analysisManager import stage, basicSampler, optimiseRoutine

	def fun1(
		depth: Union[int, float],
		width: Union[int, float],
		**kwargs: Any,
	) -> dict[str, Union[int, float]]:
		return {'area':depth*width}
	stage1: stage = stage(fun1,args=['depth','width'],name='area')
	stageGate: Callable[[pd.Series], bool] = lambda row: row['area']>10

	def fun2(
		depth: Union[int, float],
		width: Union[int, float],
		height: Union[int, float],
		**kwargs: Any,
	) -> dict[str, Union[int, float]]:
		return {'volume':depth*width*height}
	stage2: stage = stage(fun2,args = ['depth','width','height'],gate = stageGate)
	stages: list[stage] = [stage1,stage2]
	resultTargetFunctions: dict[str, Callable[[float], float]] = {
		'volume':lambda v: -v,
		'area': lambda a: a
	}
	from numpy import inf
	targetFunction: Callable[[dict[str, Any]], Tuple[float, float]] = (
		lambda row: (row.get('volume',inf),-row.get('area',-inf))
	)
	sampler: basicSampler = basicSampler(depth=(1,10),width=(2,5),height=(0.1,8))
	optimiser: optimiseRoutine = optimiseRoutine(
		targetFunction=targetFunction,
		sampler=sampler,
	)
	runner: runManager = runManager(stages=stages,routines=[optimiser],name='testRun')