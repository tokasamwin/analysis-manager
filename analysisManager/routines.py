from analysisManager.utils import hashRow, iterableCheck
from analysisManager.sampling import definedSampler
from copy import deepcopy
from scipy.interpolate import RBFInterpolator
from math import ceil, log2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core import variable as mooVars
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.termination import get_termination
from pymoo.optimize import minimize as minimise
from pymoo.core.mixed import MixedVariableSampling
from pymoo.core.mixed import MixedVariableMating
from pymoo.core.mixed import MixedVariableDuplicateElimination
import pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

class routine(object):
	def __init__(self, sampler, configuration=None, name = None):
		if configuration:
			self.configuration = configuration
		else:
			self.configuration = {}
		self.name = name
		self.sampler = sampler
	
	def getResultPopulation(self,target: str,nLen: int | None = None) -> Population:
		if not(hasattr(self,'df')):
			raise AttributeError('No DataFrame found')
		if nLen:
			if nLen<0:
				subdf = self.df.dropna().sort_values(target).iloc[nLen:]
			else:
				subdf = self.df.dropna().sort_values(target).iloc[:nLen]
			initialGuesses = [row.to_dict() for index,row in subdf.iterrows()]
			initialGuesses = [{key:d[key] for key in self.sampler.variables} for d in initialGuesses]
			pop = Population.new(X=initialGuesses)
		elif hasattr(self,'result'):
			x = self.result.X
			if type(x)==dict:
				pop = Population.new(X = [{key:x[key] for key in self.sampler.variables}])
			else:
				dictList = [{key:xi[key] for key in self.sampler.variables} for xi in x]
				pop = Population.new(X = dictList)
		return pop

	def _evaluate_single(self, index, row, runFunction, runText):
		result = runFunction(row, runText=runText)
		if result is None:
			return index, {}
		return index, result

	def _parallel_run(self, rows, runFunction, workers=None, text_template=None):
		if not workers:
			results = []
			for index, row in rows:
				runText = None if text_template is None else text_template(index, len(rows))
				results.append(self._evaluate_single(index, row, runFunction, runText))
			return results

		max_workers = len(workers) if isinstance(workers, (list, tuple, set)) else workers
		if max_workers is None or max_workers < 1:
			max_workers = 1

		futures = []
		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			for index, row in rows:
				runText = None if text_template is None else text_template(index, len(rows))
				futures.append(
					executor.submit(self._evaluate_single, index, row, runFunction, runText)
				)

			return [future.result() for future in as_completed(futures)]

class searchRoutine(routine):
	def __init__(self, sampler, configuration=None, name=None, **kwargs):
		super().__init__(sampler, configuration = configuration, name = name)
		self.df = self.sampler.getDF(**kwargs)
		self.df['ID'] = self.df.apply(hashRow,axis=1,**self.configuration)
		method = kwargs.get('method','grid')
		density = kwargs.get('density',5)
		if not(name):
			self.name = 'Search routine ({} at {} Density)'.format(method.lower(),density)
	
	def __call__(self,runFunction,workers=None,**kwargs):
		rows = list(self.df.iterrows())

		def text_template(index, count):
			return 'Search: running {} of {}'.format(index + 1, count)

		results = self._parallel_run(
			rows,
			runFunction,
			workers=workers,
			text_template=text_template,
		)

		for index, row_result in results:
			for key in row_result:
				self.df.at[index, key] = row_result[key]

		return self.df
		for index,row in self.df.iterrows():
			runText = 'Search: running {} of {}'.format(index+1,len(self.df))
			results = runFunction(row, runText=runText)
			for key in results:
				self.df.at[index, key] = results[key]
		return self.df

class GAProblem(ElementwiseProblem):
	def __init__(self, variables, nObj, runFunction, targetFunction, configuration):
		super().__init__(vars=variables, n_obj=nObj)
		self.nObj = nObj
		self.runFunction = runFunction
		self.targetFunction = targetFunction
		self.configuration = configuration
		self._evaluationsRows = []

	def _evaluate(self, row, out, *args, **kwargs):
		row = dict(row)
		row["ID"] = hashRow(row, **self.configuration)
		result = self.runFunction(
			pd.Series(row), 
			runText="{} Objective Optimisation: ".format(self.nObj)
		)
		row.update(result)
		self._evaluationsRows.append(row.copy())
		out["F"] = self.targetFunction(row)

class optimiseRoutine(routine):
	def __init__(self, sampler, targetFunction, name=None, configuration=None, **kwargs):
		"""
		This is an optimisation routine run across the sampler design space. 
		The routine will run the design space and store the results in a pd.DataFrame, which can then be used to find the optimal design.
		targetFunction is a function that retrieves output values from the rows of the DataFrame. Notably, targetFunction should use "getattr" to access values and return a default value if the attribute is not present. This is to ensure that the routine can run across the entire design space, even if some rows do not have all the required attributes. The routine will then use this targetFunction to evaluate the performance of each design and store the results in a pd.DataFrame. The pd.DataFrame can then be used to find the optimal design based on the targetFunction's output. This method allows the original values to be stored in the DataFrame and be appropriately maximised or minimised,
		E.g. targetFunction = -row.getattr('tbr',-np.inf) to maximise TBR with a default value of -infinity
		"""
		super().__init__(sampler = sampler, configuration = configuration, name = name)
		self.targetFunction = targetFunction
		exampleOutput = targetFunction({})
		if hasattr(exampleOutput,'__len__'):
			self.nObj = len(exampleOutput)
		else:
			self.nObj = 1
		self.df = pd.DataFrame(columns = self.sampler.getDF(density=1).columns)
		if type(sampler) == definedSampler:
			self.variables={}
			for var in sampler.variables:
				if var in sampler.continuousVariableNames:
					self.variables[var] = mooVars.Real(bounds=sampler.variables[var].limits)
				elif var in sampler.discreteVariableNames:
					self.variables[var] = mooVars.Choice(options = sampler.variables[var].vals)
		else:
			self.variables = {var:mooVars.Real(bounds=sampler.variables[var]) for var in sampler.variables}
		self.routineConfiguration=kwargs
		if not(name):
			self.name = '{} Variable Optimisation'.format(self.nObj)

	def __call__(self,function,workers=None,**kwargs):
		self.routineConfiguration.update(kwargs)
		self.problem = GAProblem(
			variables = self.variables,
			runFunction = function,
			targetFunction = self.targetFunction,
			configuration=self.configuration,
			nObj=self.nObj
			)
		algorithm = self.routineConfiguration.get('algorithm',NSGA2)
		survival = self.routineConfiguration.get('survival',None)
		if not(survival):
			if self.nObj==1:
				survival = FitnessSurvival()
			else:
				survival = RankAndCrowdingSurvival()
		else:
			survival = survival()
		seed=kwargs.get('seed',1)

		sampling = self.routineConfiguration.get('sampling',MixedVariableSampling())
		self.algorithm = algorithm(
			pop_size = self.routineConfiguration.get('pop_size',20),
			n_offsprings = self.routineConfiguration.get('n_offsprings',10),
			survival = survival,
			sampling = sampling,
			mating = MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
			eliminate_duplicates = MixedVariableDuplicateElimination()
		)
		terminationDefault = get_termination("n_gen", self.routineConfiguration.get('n_gen',20))
		mooKwargs = {
			'seed':seed,
			'save_history':True,
			'workers':max(int(workers),1)
		}
		termination = self.routineConfiguration.get('termination',terminationDefault)
		self.result = minimise(
			self.problem,
			self.algorithm,
			termination,
			**mooKwargs
			)
		self.df = pd.DataFrame(self.problem._evaluationsRows)
		return self.df

class surrogateRoutine(routine):
	"""Evaluate sampler points and fit a surrogate to named result columns.

	Error tolerances are checked against evaluated holdout points. When supplied,
	all configured tolerances must pass for the routine to stop early. A custom
	model may be an estimator class/instance with ``fit`` and ``predict``, or a
	callable accepting ``(X, Y)`` and returning a predictor.
	"""
	def __init__(self, sampler, targetOutputs, startDensity = 7,
			aTol=None, rTol=None, model=None,
			name=None, configuration=None, method='sobol', validation_fraction=0.2,
			error_check_interval=5):
		super().__init__(sampler, configuration=configuration, name=name)
		if isinstance(targetOutputs, str):
			targetOutputs = [targetOutputs]
		self.targetOutputs = list(targetOutputs)
		if not self.targetOutputs:
			raise ValueError('At least one target output is required')
		if startDensity < 1:
			raise ValueError('max_points must be positive')
		if aTol is not None and aTol < 0:
			raise ValueError('aTol must be non-negative')
		if rTol is not None and rTol < 0:
			raise ValueError('rTol must be non-negative')
		if not 0 < validation_fraction < 1:
			raise ValueError('validation_fraction must be between 0 and 1')
		if error_check_interval < 1:
			raise ValueError('error_check_interval must be positive')
		self.density = startDensity
		self.aTol = aTol
		self.rTol = rTol
		self.modelFactory = model
		self.validation_fraction = validation_fraction
		self.error_check_interval = error_check_interval
		self.inputNames = list(sampler.variables)
		self.method = method.lower()
		if not name:
			self.name = 'Surrogate Model ({})'.format(', '.join(self.targetOutputs))
		self.df = pd.DataFrame(columns=self.inputNames + self.targetOutputs + ['ID'])
		self.surrogate = None
		self.error = None
		self._prepareCandidates()

	def _prepareCandidates(self):
		if self.method == 'sobol':
			candidates = self.sampler.getDF(density=self.density, method='sobol')
		else:
			continuousCount = len(getattr(self.sampler, 'continuousVariableNames', self.inputNames))
			discreteCount = len(getattr(self.sampler, 'discreteSamples', [None]))
			candidates = self.sampler.getDF(density=self.density, method=self.method)
		self.candidates = candidates.loc[:, self.inputNames].reset_index(drop=True)
		if self.candidates.empty:
			raise ValueError('The sampler did not generate any candidates')
		self._numericInputs = [
			name for name in self.inputNames
			if pd.api.types.is_numeric_dtype(self.candidates[name])
		]
		self._categoricalInputs = [name for name in self.inputNames if name not in self._numericInputs]
		self._numericBounds = {}
		for name in self._numericInputs:
			low = self.candidates[name].min()
			span = self.candidates[name].max() - low
			self._numericBounds[name] = (low, span if span else 1)
		encoded = self._encodeInputs(self.candidates)
		self._featureNames = list(encoded.columns)

	def _encodeInputs(self, inputs):
		frame = pd.DataFrame(inputs).loc[:, self.inputNames].copy()
		for name in self._numericInputs:
			low, span = self._numericBounds[name]
			frame[name] = (frame[name] - low) / span
		return pd.get_dummies(frame, columns=self._categoricalInputs, dtype=float)

	def _features(self, inputs):
		encoded = self._encodeInputs(inputs).reindex(columns=self._featureNames, fill_value=0)
		return encoded.to_numpy(dtype=float)

	def _fitModel(self, inputs, outputs):
		features = self._features(inputs)
		uniqueFeatures, uniqueIndices = np.unique(features, axis=0, return_index=True)
		outputs = outputs[uniqueIndices]
		if self.modelFactory is None:
			return RBFInterpolator(uniqueFeatures, outputs, kernel='thin_plate_spline', degree=1)
		if isinstance(self.modelFactory, type):
			model = self.modelFactory()
		elif hasattr(self.modelFactory, 'fit'):
			model = deepcopy(self.modelFactory)
		else:
			return self.modelFactory(uniqueFeatures, outputs)
		model.fit(uniqueFeatures, outputs)
		return model

	@staticmethod
	def _modelPredict(model, features):
		if hasattr(model, 'predict'):
			return np.asarray(model.predict(features))
		return np.asarray(model(features))

	def _validationErrors(self):
		valid = self.df[self.targetOutputs].notna().all(axis=1).to_numpy()
		indices = np.arange(len(self.df))
		spacing = max(2, int(round(1 / self.validation_fraction)))
		validationMask = (indices + 1) % spacing == 0
		trainMask = ~validationMask
		trainMask &= valid
		validationMask &= valid
		if trainMask.sum() < 2 or validationMask.sum() < 2:
			return None
		trainInputs = self.df.loc[trainMask, self.inputNames]
		trainOutputs = self.df.loc[trainMask, self.targetOutputs].to_numpy(dtype=float)
		validationInputs = self.df.loc[validationMask, self.inputNames]
		actual = self.df.loc[validationMask, self.targetOutputs].to_numpy(dtype=float)
		model = self._fitModel(trainInputs, trainOutputs)
		predicted = self._modelPredict(model, self._features(validationInputs))
		predicted = predicted.reshape(actual.shape)
		absoluteError = np.abs(predicted - actual)
		relativeError = absoluteError / np.maximum(np.abs(actual), np.finfo(float).eps)
		return {
			'absolute': float(np.max(absoluteError)),
			'relative': float(np.max(relativeError)),
		}

	def _tolerancesMet(self, errors):
		if errors is None or (self.aTol is None and self.rTol is None):
			return False
		return (
			(self.aTol is None or errors['absolute'] <= self.aTol)
			and (self.rTol is None or errors['relative'] <= self.rTol)
		)

	def __call__(self, runFunction, workers=None):
		self.df = self.candidates.head(0).copy()
		rows = list(self.candidates.iterrows())

		def text_template(index, count):
			return 'Surrogate: running {} of {}'.format(index + 1, count)

		results = self._parallel_run(
			rows,
			runFunction,
			workers=workers,
			text_template=text_template,
		)

		for index, row_result in results:
			row = self.candidates.iloc[index].to_dict()
			row['ID'] = hashRow(pd.Series(row), **self.configuration)
			row.update(row_result)
			self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

			if ((self.aTol is not None or self.rTol is not None)
					and len(self.df) >= self.error_check_interval
					and len(self.df) % self.error_check_interval == 0):
				self.error = self._validationErrors()
				if self._tolerancesMet(self.error):
					break

		if self.aTol is not None or self.rTol is not None:
			self.error = self._validationErrors()

		valid = self.df[self.targetOutputs].notna().all(axis=1)
		if valid.sum() >= 2:
			self.surrogate = self._fitModel(
				self.df.loc[valid, self.inputNames],
				self.df.loc[valid, self.targetOutputs].to_numpy(dtype=float),
			)
		return self.df
		self.df = self.candidates.head(0).copy()
		for index, inputs in self.candidates.iterrows():
			row = inputs.to_dict()
			row['ID'] = hashRow(pd.Series(row), **self.configuration)
			runText = 'Surrogate: running {} of {}'.format(index + 1, len(self.candidates))
			result = runFunction(pd.Series(row), runText=runText)
			row.update(result)
			self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
			if ((self.aTol is not None or self.rTol is not None)
					and len(self.df) >= self.error_check_interval
					and len(self.df) % self.error_check_interval == 0):
				self.error = self._validationErrors()
				if self._tolerancesMet(self.error):
					break
		if self.aTol is not None or self.rTol is not None:
			self.error = self._validationErrors()
		valid = self.df[self.targetOutputs].notna().all(axis=1)
		if valid.sum() >= 2:
			self.surrogate = self._fitModel(
				self.df.loc[valid, self.inputNames],
				self.df.loc[valid, self.targetOutputs].to_numpy(dtype=float),
			)
		return self.df

	def predict(self, inputs):
		if self.surrogate is None:
			raise RuntimeError('Run the routine successfully before predicting')
		isSinglePoint = isinstance(inputs, dict) and not any(
			isinstance(value, (list, tuple, np.ndarray, pd.Series)) for value in inputs.values()
		)
		frame = pd.DataFrame([inputs]) if isSinglePoint else pd.DataFrame(inputs)
		predictions = self._modelPredict(self.surrogate, self._features(frame))
		predictions = predictions.reshape(len(frame), len(self.targetOutputs))
		return pd.DataFrame(predictions, columns=self.targetOutputs, index=frame.index)

if __name__ == '__main__':
	from analysisManager import sampling
	from analysisManager.stages import stage
	from analysisManager.runManager import runManager
	sampler = sampling.basicSampler(
		height=(0.1,10),
		width=(0.1,10),
		depth=(0.1,10),
	)
	from numpy import inf
	search = searchRoutine(sampler)
	def targetOptimise(row):
		return -row.get('volume',-inf),row.get('area',inf)
	MOO = optimiseRoutine(sampler,targetFunction=targetOptimise)
	from pymoo.core.mixed import MixedVariableGA
	def volumeOptimise(row):
		return -row.get('volume',-inf)
	routines = [search,MOO]
	def findArea(depth,width,**kwargs):
		return {'area':depth*width}
	areaStage = stage(findArea,inputs=['depth','width'],name = 'Area')

	maxArea = lambda row: row.get('area')<5

	def findVolume(height,width,depth,**kwargs):
		return {'volume':depth*width*height}

	volumeStage = stage(findVolume,inputs=['height','width','depth'],name='Volume',gate = maxArea)
	stages = [areaStage,volumeStage]
	
	runner = runManager(stages,routines = routines,name='Test')
	runner.runRoutines()
	
	bestResults = runner.df.dropna().sort_values('volume').iloc[-10:]
	initialGuesses = [row.to_dict() for index,row in bestResults.iterrows()]
	initialGuesses = [{key:d[key] for key in sampler.variables} for d in initialGuesses]
	population = Population.new(X=initialGuesses)
	SOO = optimiseRoutine(
		sampler,
		targetFunction = volumeOptimise,
		algorithm = MixedVariableGA,
		sourceDF = bestResults,
		sampling = population,
		)
	runner.runRoutine(SOO)
	def function(height,area,density,**kwargs):
		results = {
			'volume': height*area,
			'mass': density * height * area
		}
		return results
	runStage = stage(function,name='test',inputs = ['height','area','density'])
	localSampler = sampling.basicSampler(
		height = (10,30),
		area = (100,200),
		density = (0.01,0.04)
	)
	targetOutputs = ('volume','mass')
	surrogate = surrogateRoutine(sampler = localSampler,targetOutputs = targetOutputs)
	surrogateRunner = runManager(stages = [runStage],name='Surrogate test',routines = [surrogate])
	surrogateRunner.runRoutines()
	guessPoint = {'area':120,'height':15,'density':0.03}
	guessResult = surrogateRunner.routines[0].predict(guessPoint)