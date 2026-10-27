"""Search, optimise, or build surrogate models over sampler design spaces.

The routines in this module evaluate design points with a user-supplied
function. ``searchRoutine`` evaluates the sampler's generated points,
``optimiseRoutine`` uses a genetic algorithm to find points that minimise a
target function, and ``surrogateRoutine`` fits a model to evaluated points for
later prediction. All routines return their evaluated results as a pandas
DataFrame and can use either a worker count or an executor for parallel runs.
"""

from analysisManager.utils import hashRow
from analysisManager.sampling import definedSampler
from copy import deepcopy
from scipy.interpolate import RBFInterpolator
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
from collections.abc import Callable as _Callable, Hashable as _Hashable, Sequence as _Sequence
from typing import Any as _Any, Optional as _Optional, Union as _Union

class routine(object):
	"""Base class for routines that evaluate points from a sampler.

	Args:
		sampler: Object that provides the design variables and sampling methods.
		configuration: Optional settings passed to ``hashRow`` when assigning
			design-point IDs.
		name: Optional display name for the routine.
	"""

	def __init__(
		self,
		sampler: _Any,
		configuration: _Optional[dict[str, _Any]] = None,
		name: _Optional[str] = None,
	) -> None:
		if configuration:
			self.configuration = configuration
		else:
			self.configuration = {}
		self.name = name
		self.sampler = sampler
	
	def getResultPopulation(self,target: str,nLen: int | None = None) -> Population:
		"""Create a pymoo population from evaluated results.

		Args:
			target: DataFrame column used to sort candidate results.
			nLen: Number of sorted rows to include. A negative value selects
				rows from the end of the sorted DataFrame.

		Returns:
			A pymoo ``Population`` containing the selected sampler variables.

		Raises:
			AttributeError: If this routine has no evaluated DataFrame.
		"""
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

	def _evaluate_single(
		self,
		index: int,
		row: _Union[pd.Series, dict[str, _Any]],
		runFunction: _Callable[..., _Optional[dict[str, _Any]]],
		runText: _Optional[str],
	) -> tuple[int, dict[str, _Any]]:
		result = runFunction(row, runText=runText)
		if result is None:
			return index, {}
		return index, result

	def _parallel_run(
		self,
		rows: _Sequence[tuple[int, _Union[pd.Series, dict[str, _Any]]]],
		runFunction: _Callable[..., _Optional[dict[str, _Any]]],
		workers: _Any = None,
		text_template: _Optional[_Callable[[int, int], str]] = None,
	) -> list[tuple[int, dict[str, _Any]]]:
		workerIntVal = getattr(workers,'__int__',lambda: 1)()
		workerCheck = [
			workerIntVal>1,
			type(workers) in [ProcessPoolExecutor,ThreadPoolExecutor]
		]
		if not(any(workerCheck)):
			results = []
			for index, row in rows:
				runText = None if text_template is None else text_template(index, len(rows))
				results.append(self._evaluate_single(index, row, runFunction, runText))
			return results

		futures = []
		if workerCheck[1]:
			for index,row in rows:
				runText = None if text_template is None else text_template(index, len(rows))
				futures.append(
					workers.submit(self._evaluate_single, index, row, runFunction, runText)
				)
			return [future.result() for future in as_completed(futures)]

		with ProcessPoolExecutor(max_workers = workerIntVal) as executor:
			for index, row in rows:
				runText = None if text_template is None else text_template(index, len(rows))
				futures.append(
					executor.submit(self._evaluate_single, index, row, runFunction, runText)
				)
		return [future.result() for future in as_completed(futures)]

class searchRoutine(routine):
	"""Evaluate every design point generated by a sampler.

	The generated points and their IDs are available in ``df``. Calling the
	routine evaluates each point with ``runFunction`` and adds the returned
	values as columns.

	Args:
		sampler: Sampler used to generate the design points.
		configuration: Optional settings used when generating point IDs.
		name: Optional display name.
		**kwargs: Options forwarded to the sampler's ``getDF`` method.
	"""

	def __init__(
		self,
		sampler: _Any,
		configuration: _Optional[dict[str, _Any]] = None,
		name: _Optional[str] = None,
		**kwargs: _Any,
	) -> None:
		super().__init__(sampler, configuration = configuration, name = name)
		self.df = self.sampler.getDF(**kwargs)
		self.df['ID'] = self.df.apply(hashRow,axis=1,**self.configuration)
		method = kwargs.get('method','grid')
		density = kwargs.get('density',5)
		if not(name):
			self.name = 'Search routine ({} at {} Density)'.format(method.lower(),density)
	
	def __call__(
		self,
		runFunction: _Callable[..., _Optional[dict[str, _Any]]],
		workers: _Any = None,
		**kwargs: _Any,
	) -> pd.DataFrame:
		"""Evaluate the sampled design points.

		Args:
			runFunction: Function called once per point. It receives the point
				and an optional ``runText`` progress string, and may return a
				dictionary of result values.
			workers: Worker count, ``ThreadPoolExecutor``, or
				``ProcessPoolExecutor`` used to run evaluations concurrently.
			**kwargs: Accepted for compatibility; currently unused.

		Returns:
			The routine's DataFrame, including inputs, IDs, and evaluation
			results.
		"""
		rows = list(self.df.iterrows())

		def text_template(index: int, count: int) -> str:
			return 'Search: running {} of {}'.format(index + 1, count)

		results = self._parallel_run(
			rows,
			runFunction,
			workers=workers,
			text_template=text_template,
		)

		for index, row_result in results:
			for key, value in row_result.items():
				if key not in self.df.columns:
					self.df[key] = pd.Series(index=self.df.index, dtype=object)
				self.df.at[index, key] = value

		return self.df

class GAProblem(ElementwiseProblem):
	"""Adapt a user evaluation function to pymoo's elementwise problem API.

	The class hashes each candidate, runs the evaluation function, records
	evaluated rows, and passes the target function's output to pymoo as the
	objective vector.
	"""

	def __init__(
		self,
		variables: dict[str, _Any],
		nObj: int,
		runFunction: _Callable[..., dict[str, _Any]],
		targetFunction: _Callable[[dict[str, _Any]], _Any],
		configuration: dict[str, _Any],
	) -> None:
		super().__init__(vars=variables, n_obj=nObj)
		self.nObj = nObj
		self.runFunction = runFunction
		self.targetFunction = targetFunction
		self.configuration = configuration
		self._evaluationsRows = []

	def _evaluate(
		self,
		row: dict[str, _Any],
		out: dict[str, _Any],
		*args: _Any,
		**kwargs: _Any,
	) -> None:
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
	def __init__(
		self,
		sampler: _Any,
		targetFunction: _Callable[[dict[str, _Any]], _Any],
		name: _Optional[str] = None,
		configuration: _Optional[dict[str, _Any]] = None,
		**kwargs: _Any,
	) -> None:
		"""Optimise a target function over the sampler's design space.

		The target function receives each evaluated row as a dictionary and
		returns one objective value or a sequence of objective values. pymoo
		minimises objectives, so negate values that should be maximised. Rows
		evaluated by the optimiser are stored in ``df``.

		Args:
			sampler: Sampler defining the optimisation variables and bounds.
			targetFunction: Function that extracts objective values from a
				row. It should handle missing result keys, for example with
				``row.get("value", default)``.
			name: Optional display name.
			configuration: Optional settings used when hashing evaluated
				points.
			**kwargs: Optimiser options such as ``algorithm``, ``pop_size``,
				``n_offsprings``, ``n_gen``, ``sampling``, and ``termination``.

		Example:
			Negate a metric to maximise it while retaining its original value
			in the result DataFrame: ``lambda row: -row.get("tbr", -np.inf)``.
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

	def __call__(
		self,
		function: _Callable[..., dict[str, _Any]],
		workers: _Any = None,
		**kwargs: _Any,
	) -> pd.DataFrame:
		"""Run the configured genetic optimisation.

		Args:
			function: Function that evaluates one candidate and returns a
				dictionary of result values.
			workers: Number of pymoo evaluation workers. Values below one are
				clamped to one.
			**kwargs: Per-run optimisation settings, which override settings
				provided at construction.

		Returns:
			A DataFrame containing the evaluated candidates and their results.
		"""
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
		if workers == None:
			workers = 1
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

	Args:
		sampler: Sampler used to generate candidate inputs.
		targetOutputs: Name or names of result columns to model.
		startDensity: Density passed to the sampler to generate candidates.
		aTol: Optional maximum absolute validation error for early stopping.
		rTol: Optional maximum relative validation error for early stopping.
		model: Optional custom model class, fitted model instance, or callable
			that creates a predictor from feature and output arrays.
		name: Optional display name.
		configuration: Optional settings used when hashing points.
		method: Sampling method passed to the sampler.
		validation_fraction: Fraction of valid evaluations reserved for
			validation.
		error_check_interval: Number of evaluations between tolerance checks.
	"""
	def __init__(
		self,
		sampler: _Any,
		targetOutputs: _Union[str, _Sequence[str]],
		startDensity: int = 7,
		aTol: _Optional[float] = None,
		rTol: _Optional[float] = None,
		model: _Any = None,
		name: _Optional[str] = None,
		configuration: _Optional[dict[str, _Any]] = None,
		method: str = 'sobol',
		validation_fraction: float = 0.2,
		error_check_interval: int = 5,
	) -> None:
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

	def _prepareCandidates(self) -> None:
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

	def _encodeInputs(self, inputs: pd.DataFrame) -> pd.DataFrame:
		frame = pd.DataFrame(inputs).loc[:, self.inputNames].copy()
		for name in self._numericInputs:
			low, span = self._numericBounds[name]
			frame[name] = (frame[name] - low) / span
		return pd.get_dummies(frame, columns=self._categoricalInputs, dtype=float)

	def _features(self, inputs: pd.DataFrame) -> np.ndarray:
		encoded = self._encodeInputs(inputs).reindex(columns=self._featureNames, fill_value=0)
		return encoded.to_numpy(dtype=float)

	def _fitModel(self, inputs: pd.DataFrame, outputs: np.ndarray) -> _Any:
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
	def _modelPredict(model: _Any, features: np.ndarray) -> np.ndarray:
		if hasattr(model, 'predict'):
			return np.asarray(model.predict(features))
		return np.asarray(model(features))

	def _validationErrors(self) -> _Optional[dict[str, float]]:
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

	def _tolerancesMet(self, errors: _Optional[dict[str, float]]) -> bool:
		if errors is None or (self.aTol is None and self.rTol is None):
			return False
		return (
			(self.aTol is None or errors['absolute'] <= self.aTol)
			and (self.rTol is None or errors['relative'] <= self.rTol)
		)

	def __call__(
		self,
		runFunction: _Callable[..., _Optional[dict[str, _Any]]],
		workers: _Any = None,
	) -> pd.DataFrame:
		"""Evaluate candidate points and fit the surrogate model.

		Args:
			runFunction: Function called for each point. It receives the point
				and an optional ``runText`` progress string, and may return a
				dictionary of result values.
			workers: Worker count, ``ThreadPoolExecutor``, or
				``ProcessPoolExecutor`` used to run evaluations concurrently.

		Returns:
			A DataFrame containing candidate inputs, IDs, and evaluation
			results. Successfully evaluated target outputs are used to fit the
			surrogate stored in ``surrogate``.
		"""
		self.df = self.candidates.head(0).copy()
		self.candidates['ID'] = self.candidates.apply(hashRow,axis=1,**self.configuration)
		rows = list(self.candidates.iterrows())

		def text_template(index: int, count: int) -> str:
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

	def predict(self, inputs: _Any) -> pd.DataFrame:
		"""Predict target outputs for one or more input points.

		Args:
			inputs: A dictionary for one point or a DataFrame-compatible
				collection of points, with columns matching the sampler inputs.

		Returns:
			A DataFrame with one column per target output and the input row
			index.

		Raises:
			RuntimeError: If the routine has not yet fitted a surrogate.
		"""
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
	sampler: sampling.basicSampler = sampling.basicSampler(
		height=(0.1,10),
		width=(0.1,10),
		depth=(0.1,10),
	)
	from numpy import inf
	search: searchRoutine = searchRoutine(sampler)
	def targetOptimise(row: dict[str, _Any]) -> tuple[float, float]:
		return -row.get('volume',-inf),row.get('area',inf)
	MOO: optimiseRoutine = optimiseRoutine(sampler,targetFunction=targetOptimise)
	from pymoo.core.mixed import MixedVariableGA
	def volumeOptimise(row: dict[str, _Any]) -> float:
		return -row.get('volume',-inf)
	routines: list[routine] = [search,MOO]
	def findArea(depth: float, width: float, **kwargs: _Any) -> dict[str, float]:
		return {'area':depth*width}
	areaStage: stage = stage(findArea,inputs=['depth','width'],name = 'Area')

	#maxArea = lambda row: row.get('area')<5
	def maxArea(row: dict[str, _Any]) -> bool: return row.get('area')<5
	def findVolume(
		height: float,
		width: float,
		depth: float,
		**kwargs: _Any,
	) -> dict[str, float]:
		return {'volume':depth*width*height}

	volumeStage: stage = stage(findVolume,inputs=['height','width','depth'],name='Volume',gate = maxArea)
	stages: list[stage] = [areaStage,volumeStage]
	
	runner: runManager = runManager(stages,routines = routines,name='Test')
	runner.runRoutines(workers = 4)
	
	bestResults: pd.DataFrame = runner.df.dropna().sort_values('volume').iloc[-10:]
	initialGuesses: list[dict[_Hashable, _Any]] = [row.to_dict() for index,row in bestResults.iterrows()]
	initialGuesses: list[dict[_Hashable, _Any]] = [{key:d[key] for key in sampler.variables} for d in initialGuesses]
	population: Population = Population.new(X=initialGuesses)
	SOO: optimiseRoutine = optimiseRoutine(
		sampler,
		targetFunction = volumeOptimise,
		algorithm = MixedVariableGA,
		sourceDF = bestResults,
		sampling = population,
		)
	runner.runRoutine(SOO)
	def function(
		height: float,
		area: float,
		density: float,
		**kwargs: _Any,
	) -> dict[str, float]:
		results = {
			'volume': height*area,
			'mass': density * height * area
		}
		return results
	runStage: stage = stage(function,name='test',inputs = ['height','area','density'])
	localSampler: sampling.basicSampler = sampling.basicSampler(
		height = (10,30),
		area = (100,200),
		density = (0.01,0.04)
	)
	targetOutputs: tuple[str, str] = ('volume','mass')
	surrogate: surrogateRoutine = surrogateRoutine(sampler = localSampler,targetOutputs = targetOutputs)
	surrogateRunner: runManager = runManager(stages = [runStage],name='Surrogate test',routines = [surrogate])
	with ProcessPoolExecutor(max_workers = 8) as mp_pool:
		surrogateRunner.runRoutines(workers = mp_pool)
	guessPoint: dict[str, float] = {'area':120,'height':15,'density':0.03}
	guessResult: pd.DataFrame = surrogateRunner.routines[0].predict(guessPoint)