import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysisManager.routines import (
    GAProblem,
    optimiseRoutine,
    routine,
    searchRoutine,
    surrogateRoutine,
)
from analysisManager.sampling import basicSampler


class MeanModel:
    def fit(self, features, outputs):
        self.mean = np.mean(outputs, axis=0)
        return self

    def predict(self, features):
        return np.tile(self.mean, (len(features), 1))


class RoutineTests(unittest.TestCase):
    def setUp(self):
        self.sampler = basicSampler(x=(0.0, 1.0))

    def test_evaluate_single_converts_none_to_empty_result(self):
        subject = routine(self.sampler)

        result = subject._evaluate_single(
            3,
            pd.Series({"x": 0.5}),
            lambda row, runText: None,
            "progress",
        )

        self.assertEqual(result, (3, {}))

    def test_parallel_run_calls_serial_function_with_progress_text(self):
        subject = routine(self.sampler)
        rows = [(0, {"x": 0.0}), (1, {"x": 1.0})]
        calls = []

        def run(row, runText):
            calls.append((row["x"], runText))
            return {"y": row["x"]}

        results = subject._parallel_run(
            rows,
            run,
            text_template=lambda index, count: f"{index + 1}/{count}",
        )

        self.assertEqual(results, [(0, {"y": 0.0}), (1, {"y": 1.0})])
        self.assertEqual(calls, [(0.0, "1/2"), (1.0, "2/2")])

    def test_parallel_run_accepts_a_thread_pool(self):
        subject = routine(self.sampler)
        rows = [(0, {"x": 2}), (1, {"x": 4})]

        with ThreadPoolExecutor(max_workers=2) as workers:
            results = subject._parallel_run(
                rows, lambda row, runText: {"y": row["x"] * 2}, workers=workers
            )

        self.assertEqual(dict(results), {
            0: {"y": 4},
            1: {"y": 8},
        })

    def test_get_result_population_requires_a_dataframe(self):
        with self.assertRaisesRegex(AttributeError, "No DataFrame found"):
            routine(self.sampler).getResultPopulation("score")

    def test_get_result_population_selects_sorted_rows(self):
        subject = routine(self.sampler)
        subject.df = pd.DataFrame(
            {"x": [0.8, 0.2, 0.5], "score": [3.0, 1.0, 2.0]}
        )

        population = subject.getResultPopulation("score", nLen=2)

        self.assertEqual(
            [individual.X["x"] for individual in population],
            [0.2, 0.5],
        )

    def test_get_result_population_uses_optimization_result(self):
        subject = routine(self.sampler)
        subject.df = pd.DataFrame()
        subject.result = type("Result", (), {"X": {"x": 0.75}})()

        population = subject.getResultPopulation("score")

        self.assertEqual(len(population), 1)
        self.assertEqual(population[0].X, {"x": 0.75})


class SearchRoutineTests(unittest.TestCase):
    def test_search_evaluates_rows_and_retains_none_results(self):
        sampler = basicSampler(x=(0.0, 1.0))
        subject = searchRoutine(sampler, density=3, name="custom")
        calls = []

        def run(row, runText):
            calls.append(runText)
            if row["x"] == 0.5:
                return None
            return {"score": row["x"] * 2}

        result = subject(run)

        self.assertEqual(subject.name, "custom")
        self.assertEqual(len(result), 3)
        self.assertEqual(result["score"].iloc[0], 0.0)
        self.assertTrue(pd.isna(result["score"].iloc[1]))
        self.assertEqual(result["score"].iloc[2], 2.0)
        self.assertEqual(calls, ["Search: running 1 of 3",
                                 "Search: running 2 of 3",
                                 "Search: running 3 of 3"])
        self.assertTrue(result["ID"].notna().all())

    def test_search_generates_default_name(self):
        subject = searchRoutine(basicSampler(x=(0.0, 1.0)), density=2)

        self.assertEqual(subject.name, "Search routine (grid at 2 Density)")


class GAProblemTests(unittest.TestCase):
    def test_evaluate_adds_id_records_results_and_sets_objective(self):
        problem = GAProblem(
            variables={"x": object()},
            nObj=1,
            runFunction=lambda row, runText: {"score": row["x"] + 2},
            targetFunction=lambda row: row["score"],
            configuration={},
        )
        output = {}

        problem._evaluate({"x": 3.0}, output)

        self.assertEqual(output["F"], 5.0)
        self.assertEqual(len(problem._evaluationsRows), 1)
        self.assertEqual(problem._evaluationsRows[0]["score"], 5.0)
        self.assertTrue(problem._evaluationsRows[0]["ID"])


class OptimiseRoutineTests(unittest.TestCase):
    def test_call_configures_optimizer_and_stores_evaluated_rows(self):
        sampler = basicSampler(x=(0.0, 1.0))
        subject = optimiseRoutine(
            sampler,
            targetFunction=lambda row: row.get("score", 0.0),
            pop_size=4,
            n_offsprings=2,
            n_gen=3,
        )
        recorded = {}
        result = object()

        class RecordingAlgorithm:
            def __init__(self, **kwargs):
                recorded["algorithm_options"] = kwargs

        def fake_minimise(problem, algorithm, termination, **kwargs):
            output = {}
            problem._evaluate({"x": 0.5}, output)
            recorded.update(
                problem=problem,
                algorithm=algorithm,
                termination=termination,
                minimize_options=kwargs,
                objective=output["F"],
            )
            return result

        with patch("analysisManager.routines.minimise", side_effect=fake_minimise):
            frame = subject(
                lambda row, runText: {"score": row["x"] * 10},
                algorithm=RecordingAlgorithm,
            )

        self.assertIs(subject.result, result)
        self.assertEqual(recorded["objective"], 5.0)
        self.assertEqual(recorded["algorithm_options"]["pop_size"], 4)
        self.assertEqual(recorded["algorithm_options"]["n_offsprings"], 2)
        self.assertEqual(recorded["minimize_options"]["seed"], 1)
        self.assertEqual(recorded["minimize_options"]["workers"], 1)
        self.assertEqual(frame.loc[0, "score"], 5.0)
        self.assertTrue(frame.loc[0, "ID"])


class SurrogateRoutineTests(unittest.TestCase):
    def test_constructor_validates_configuration(self):
        sampler = basicSampler(x=(0.0, 1.0))
        invalid_options = [
            ({"targetOutputs": []}, "At least one target output"),
            ({"targetOutputs": "y", "startDensity": 0}, "max_points must be positive"),
            ({"targetOutputs": "y", "aTol": -1}, "aTol must be non-negative"),
            ({"targetOutputs": "y", "rTol": -1}, "rTol must be non-negative"),
            ({"targetOutputs": "y", "validation_fraction": 0}, "validation_fraction"),
            ({"targetOutputs": "y", "error_check_interval": 0}, "error_check_interval"),
        ]

        for options, message in invalid_options:
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, message):
                    surrogateRoutine(sampler, **options)

    def test_string_target_is_normalized_and_inputs_are_encoded(self):
        subject = surrogateRoutine(
            basicSampler(x=(0.0, 1.0)),
            targetOutputs="score",
            startDensity=2,
            method="grid",
        )

        self.assertEqual(subject.targetOutputs, ["score"])
        np.testing.assert_allclose(
            subject._features(pd.DataFrame({"x": [0.0, 0.5, 1.0]})),
            [[0.0], [0.5], [1.0]],
        )

    def test_run_fits_model_and_predicts_single_and_multiple_points(self):
        subject = surrogateRoutine(
            self._mixed_sampler(),
            targetOutputs=["score"],
            startDensity=2,
            model=MeanModel(),
        )

        result = subject(lambda row, runText: {"score": row["x"] * 2})

        self.assertEqual(len(result), 4)
        self.assertIsNotNone(subject.surrogate)
        single = subject.predict({"x": 0.5, "kind": "a"})
        multiple = subject.predict(
            {"x": [0.25, 0.75], "kind": ["a", "b"]}
        )
        self.assertEqual(list(single.columns), ["score"])
        self.assertEqual(len(single), 1)
        self.assertEqual(multiple.index.tolist(), [0, 1])
        self.assertEqual(multiple.shape, (2, 1))

    def test_predict_requires_a_successful_run(self):
        subject = surrogateRoutine(self._mixed_sampler(), targetOutputs="score")

        with self.assertRaisesRegex(RuntimeError, "Run the routine successfully"):
            subject.predict({"x": 0.5, "kind": "a"})

    def test_tolerance_check_stops_after_successful_validation(self):
        subject = surrogateRoutine(
            basicSampler(x=(0.0, 1.0)),
            targetOutputs="score",
            startDensity=4,
            aTol=0,
            model=MeanModel(),
            method="grid",
            validation_fraction=0.5,
            error_check_interval=4,
        )
        calls = []

        def run(row, runText):
            calls.append(runText)
            return {"score": 0.0}

        result = subject(run)

        self.assertEqual(len(result), 4)
        self.assertEqual(len(calls), 4)
        self.assertEqual(subject.error, {"absolute": 0.0, "relative": 0.0})

    @staticmethod
    def _mixed_sampler():
        class MixedSampler:
            variables = {"x": None, "kind": None}

            def getDF(self, density, method):
                return pd.DataFrame(
                    {
                        "x": [0.0, 0.5, 1.0, 0.25],
                        "kind": ["a", "a", "b", "b"],
                    }
                )

        return MixedSampler()


if __name__ == "__main__":
    unittest.main()
