from analysisManager import config
from analysisManager.utils import getDpInputs, hashRow
from shutil import rmtree as removeFolder
from json import dump
from pathlib import Path
dataDir = Path(config['path']['dataDir'])
from copy import copy
import pandas as pd

'''
This module contains tools to conduct analysis, including two separate tools for conducting analysis: a "searcher" class for conducting a search over pre-defined design points, and an "optimiser" class for optimising within a design space. Both of these are used to conduct analysis, which is defined by "stage" classes. Stages are used to create, run and manage multistaged analysis.

'''

class runManager(object):
	def __init__(self, stages, routines, name='Default',configuration=None,verbose=True):
		self._verbose = verbose 
		self.stages = stages
		self.routines = routines
		self.name = name
		self.df = pd.DataFrame()
		if configuration:
			self.configuration = configuration
		else:
			self.configuration = {}
		if configuration:
			self._configHash = hashRow(self.configuration)
		else:
			self._configHash = ''
		self.basePath = dataDir / self.name
		if not(self.basePath.is_dir()): self.basePath.mkdir(parents=True)
		fname = self.basePath / 'config-{}.json'.format(self._configHash)
		with open(fname,'w') as f:
			dump(self.configuration,f,indent=4)
	
	def runDP(self, row, runText=None):
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
				result = stage.function(*args,path=nwd,outFile=outFile)
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

	def runRoutine(self,routine,**kwargs):
		localDF = routine(self.runDP,**kwargs).copy(deep=True)
		localDF['Routine'] = routine.name
		self.df = pd.concat([self.df,localDF],axis=0,ignore_index=True).reset_index(drop=True)

	def runRoutines(self,**kwargs):
		for routine in self.routines:
			self.runRoutine(routine,**kwargs)
	
	def trimIDs(self,*IDs):
		for ID in IDs:
			folder = self.basePath / ID
			if folder.is_dir():
				removeFolder(folder)

if __name__ == '__main__':
	def fun1(depth,width,**kwargs):
		return {'area':depth*width}
	stage1 = stage(fun1,args=['depth','width'],name='area')
	stageGate = lambda row: ['area']>10
	def fun2(depth,width,height,**kwargs):
		return {'volume':depth*width*height}
	stage2 = stage(fun2,args = ['depth','width','height'],gate = stageGate)
	stages = [stage1,stage2]
	resultTargetFunctions = {
		'volume':lambda v: -v,
		'area': lambda a: a
	}
	sampler = dfs.sampler(depth=(1,10),width=(2,5),height=(0.1,8))
	args = list(sampler.variables.keys())
	for arg in args:
		sampler.variables['arg;'+arg] = sampler.variables[arg]
		del(sampler.variables[arg])
	optimiser = runOptimiserGA(stages,sampler,name='Code testing')
	optimiser.generateProblem([-1e6,1e6],**resultTargetFunctions)