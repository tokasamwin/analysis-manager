import numpy as np
from pandas import DataFrame as df
from scipy.stats.qmc import Sobol
from collections.abc import Iterable
sobolSeed = 1


class basicSampler(object):
	def __init__(self, **variables: dict[str,tuple[float,float]]):
		self.variables = variables
	
	def sobol(self,density: int) -> np.ndarray:
		sampler=Sobol(len(self.variables),rng=sobolSeed)
		points=sampler.random_base2(density)
		self.samples=[]
		for point in points:
			sample=[]
			for point_ii,name in zip(point,self.variables):
				var_range=self.variables[name]
				sample+=[var_range[0] + point_ii*(var_range[1]-var_range[0])]
			self.samples.append(sample)
		return np.array(self.samples)
	
	def grid(self,density: int | Iterable =5) -> list[list] | np.ndarray:
		ranges=[self.variables[name] for name in self.variables]
		if density==1:
			self.samples=[[np.mean(valRange) for valRange in ranges]]
		elif type(density) in iterClasses:
			if len(density)!=len(self.variables): raise AttributeError('Incorrect density shape: use a {} length iterable'.format(len(self.variables)))
			axes=[]
			for valDensity,valRange in zip(density,ranges):
				axes.append(np.linspace(valRange[0],valRange[1],valDensity))
			self.samples=np.array(np.meshgrid(*axes)).T.reshape(-1,len(self.variables))
		else:
			axes=[np.linspace(valRange[0],valRange[1],density) for valRange in ranges]
			self.samples=np.array(np.meshgrid(*axes)).T.reshape(-1,len(self.variables))
		return self.samples

	def getPoints(self,density: int | Iterable = 5,method: str ='grid') -> list[dict]:
		if method.lower() not in ['grid','sobol']:
			raise AttributeError('Invalid method, choose from "grid" or "sobol" (case insensitive)')
		points = getattr(self,method.lower())(density)
		pointsList = [
			{key:value for key,value in zip(self.variables,values)}
			for values in points
		]
		return pointsList
	
	def getDF(self, density: int | Iterable = 5, method: str = 'grid') -> df:
		pointDict=self.getPoints(density=density,method=method)
		return df(pointDict)
	
	def getSample(self,x0: list[float]) -> np.ndarray:
		result = np.array([
			self.variables[k][0] + (self.variables[k][1]-self.variables[k][0])*x for k,x in zip(self.variables,x0)
		])
		return result

from numpy import array

def distributedParameterGrid(parameters: int = 3 ,subdivisions: int = 5) -> np.ndarray:
	basis=[i for i in range(subdivisions)]
	results=[[v]+[None]*(parameters-1) for v in basis]
	index=1
	while index<parameters:
		linesToDelete=[]
		for lineIndent,line in enumerate(results):
			if line[index]==None:
				newLines=_inflateParameterGrid(subdivisions,line)
				results+=newLines
				linesToDelete.append(lineIndent)
		for indent in linesToDelete[::-1]: del(results[indent]) #needs to be done in reverse so that the removals don't effect later indents from moving
		index+=1
	results=[array(i)/(subdivisions-1) for i in results]
	results=list(filter(lambda i: sum(i)==1,results))
	return np.ndarray(results)

from copy import copy
_lineNoneSum=lambda array: sum(list(filter(lambda i: i,array))) #filters NoneTypes out of the line
def _inflateParameterGrid(subdivisions: int,line: list) -> list:
	budget=subdivisions - _lineNoneSum(line)
	index=line.index(None) # this works because index returns first index of the None value
	results=[]
	for i in range(budget):
		newLine=copy(line)
		newLine[index]=i
		results.append(newLine)
	return results

class _distribution(object):
	def __init__(self):
		self._index=0

	def __iter__(self):
		return iter(self.vals)

	def __next__(self):
		if self._index >=len(self.vals):
			raise StopIteration
		value = self.vals[self._index]
		self._index+=1
		return value

class setDistribution(_distribution):
	def __init__(self,vals: Iterable):
		super().__init__()
		self.vals=vals
	
	def __len__(self):
		return len(self.vals)
	
class linearDistribution(_distribution):
	def __init__(
		self,
		limits: tuple[float,float],
		density: int = 5
		):
		super().__init__()
		self.limits=limits
		self.vals=list(np.linspace(*limits,num=density))

	def __call__(self,index: float) -> float:
		return self.limits[0] + index*(self.limits[1]-self.limits[0])

class exponentialDistribution(_distribution):
	def __init__(
		self,
		limits: tuple[float,float],
		density: int = 5,
		base: float = 10
		):
		super().__init__()
		self._base=base
		self.limits=limits
		self._loglimits=[np.log(val)/np.log(base) for val in limits]
		baseVals = [self._loglimits[0] + i*(self._loglimits[1]-self._loglimits[0]) for i in np.linspace(0,1,num=density)]
		self.vals = [base**val for val in baseVals]

	def __call__(self,index: float) -> float:
		logIndex=self._loglimits[0] + index*(self._loglimits[1]-self._loglimits[0])
		return self._base**logIndex
	
allowedClasses=(setDistribution,linearDistribution,exponentialDistribution)
continuousClasses=(linearDistribution,exponentialDistribution)
discreteClasses=(setDistribution,)

class definedSampler(object):
	def __init__(self,**variables: dict[str,_distribution]):
		self.variables=variables
		errantVariables=list(filter(lambda i: type(variables[i]) not in allowedClasses,variables))
		if len(errantVariables):
			errorString='Following variables are not in the allowed class types: '
			errorString+='; '.join(['{}'.format(key) for key in errantVariables])
			errorString+='\ndataFrameSearch.definedSampler class {} uses defined classes:\n\t'
			errorString+='\n\t'.join(['dataFrameSearch.'+fun.__name__ for fun in allowedClasses])
			raise TypeError(errorString)
		self.continuousVariableNames=list(filter(
			lambda key: type(self.variables[key]) in continuousClasses,
			self.variables
			))
		self.discreteVariableNames=list(filter(
			lambda key: type(self.variables[key]) in discreteClasses, 
			self.variables
			))
		discreteGrid=np.meshgrid(*[list(self.variables[name]) for name in self.discreteVariableNames],indexing='ij')
		discreteSamplesArray=np.array([grid.flatten() for grid in discreteGrid]).T.tolist()
		self.discreteSamples=[{name:value for name,value in zip(self.discreteVariableNames,sample)} for sample in discreteSamplesArray]

	def sobol(self,density: int) -> list[dict]:
		sampler=Sobol(len(self.continuousVariableNames))
		points=sampler.random_base2(density)
		continuousSamples=[]
		for point in points:
			sample={}
			for point_ii,name in zip(point,self.continuousVariableNames):
				sample[name]=self.variables[name](point_ii)
			continuousSamples.append(sample)
		self.samples=self._permutate(continuousSamples)
		return self.samples
	
	def grid(self,density: int | tuple[int] | list[int]) -> list[dict]:
		if density==1:
			continuousSamples=[{name:self.variables[name](0.5) for name in self.continuousVariableNames}] # median value in the range
		elif type(density) in iterClasses:
			if len(density)!=len(self.continuousVariableNames): raise AttributeError('Incorrect density shape')
			axes=[]
			for val in density:
				if val==1:
					axes.append(np.array([0.5]))
				else:
					axes.append(np.linspace(0,1,val))
			varRanges=[[self.variables[name](val) for val in axis] for name,axis in zip(self.continuousVariableNames,axes)]
			grids=np.meshgrid(*varRanges,indexing='ij') #permutate the ranges
			#
			continuousSamples=np.array([grid.flatten() for grid in grids]).T.tolist()
			continuousSamples=[{name:value for name,value in zip(self.continuousVariableNames,values)} for values in continuousSamples]
		else:
			axes=[np.linspace(0,1,density) for name in self.continuousVariableNames]
			varRanges=[[self.variables[name](val) for val in axis] for name,axis in zip(self.continuousVariableNames,axes)]
			grids=np.meshgrid(*varRanges,indexing='ij')
			continuousSamples=np.array([grid.flatten() for grid in grids]).T.tolist()
			continuousSamples=[{name:value for name,value in zip(self.continuousVariableNames,values)} for values in continuousSamples]
		self.samples=self._permutate(continuousSamples)
		return self.samples

	def getPoints(self,density: int | Iterable, method='grid') -> list[dict]:
		if method not in ['grid','sobol']:
			raise AttributeError('Invalid method')
		points=getattr(self,method)(density)
		return points
	
	def getDF(self,density: int | Iterable = 5, method: str = 'grid') -> df:
		return df(self.getPoints(density=density,method=method))

	
	def _permutate(self,continuousSamples: list) -> list:
		if not(self.discreteSamples): return continuousSamples
		result=[]
		for cSample in continuousSamples:
			for dSample in self.discreteSamples:
				newSample=cSample | dSample
				result.append(newSample)
		return result

	def importPermutate(self,importedSamples: list) -> list:
		result=[]
		self.importedSamples=importedSamples
		for iSample in self.samples:
			for oSample in importedSamples:
				newSample=iSample | oSample
				result.append(newSample)
		self.samples=result
		return self.samples

if __name__=='__main__':
	moderators=setDistribution(['H2O','D2O','isotropic graphite HPG-59','Beryllium'])
	breeders=setDistribution(['Li4SiO4'])
	fissiles=setDistribution(['Uranium Dioxide','Uranium-Plutonium, Mixed Oxide (MOX)'])
	fissileEnrichment=linearDistribution((0.8,20),density=5)
	breederEnrichment=linearDistribution((4.5,90),density=5)
	example=definedSampler(breeders=breeders,fissiles=fissiles,moderators=moderators,fissileEnrichment=fissileEnrichment,breederEnrichment=breederEnrichment)
	exampleDF = example.getDF(method='grid',density=1)
	basicExample = basicSampler(
		fissileEnrichment = (0.8,20),
		breederEnrichment = (4.5,90)
		)
	basicDF = basicExample.getDF()