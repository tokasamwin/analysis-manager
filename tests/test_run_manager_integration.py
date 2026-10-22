import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from analysisManager import basicSampler, runManager, searchRoutine, stage


class RunManagerIntegrationTests(unittest.TestCase):
    def test_run_routines_executes_stages_and_persists_results(self):
        run_manager_module = importlib.import_module("analysisManager.runManager")
        stage_calls = []

        def calculate_double(x, **kwargs):
            stage_calls.append(("double", x))
            return {"doubled": x * 2}

        def calculate_score(x, doubled, **kwargs):
            stage_calls.append(("score", x))
            return {"score": x + doubled}

        stages = [
            stage(calculate_double, inputs=["x"], name="double"),
            stage(
                calculate_score,
                inputs=["x", "doubled"],
                gate=lambda row: row["doubled"] > 0.5,
                name="score",
            ),
        ]
        routine = searchRoutine(
            basicSampler(x=(0.0, 1.0)), density=2, name="integration search"
        )

        with tempfile.TemporaryDirectory() as data_dir, patch.object(
            run_manager_module, "dataDir", Path(data_dir)
        ):
            manager = runManager(
                stages=stages,
                routines=[routine],
                name="integration",
                configuration={"mode": "test"},
                verbose=False,
            )
            manager.runRoutines()

            results = manager.df.set_index("x")
            self.assertTrue(pd.isna(results.loc[0.0, "doubled"]))
            self.assertEqual(results.loc[0.0, "pass"], "")
            self.assertEqual(results.loc[1.0, "doubled"], 2.0)
            self.assertEqual(results.loc[1.0, "score"], 3.0)
            self.assertEqual(results.loc[1.0, "pass"], "score")
            self.assertEqual(
                stage_calls,
                [("double", 0.0), ("double", 1.0), ("score", 1.0)],
            )

            failed_id = results.loc[0.0, "ID"]
            passed_id = results.loc[1.0, "ID"]
            failed_path = manager.basePath / failed_id
            passed_path = manager.basePath / passed_id

            self.assertTrue((failed_path / "inputs.json").is_file())
            self.assertFalse((failed_path / "results.json").exists())
            self.assertTrue((passed_path / "results.json").is_file())
            with (passed_path / "inputs.json").open() as file:
                inputs = json.load(file)
            with (passed_path / "results.json").open() as file:
                persisted_results = json.load(file)

            self.assertEqual(inputs["configuration ID"], manager._configHash)
            self.assertEqual(inputs["x"], 1.0)
            self.assertEqual(persisted_results, {"doubled": 2.0, "pass": "score", "score": 3.0})
            self.assertTrue(
                (manager.basePath / f"config-{manager._configHash}.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
