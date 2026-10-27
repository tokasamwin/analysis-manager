import unittest

import numpy as np

from analysisManager.sampling import (
    basicSampler,
    definedSampler,
    distributedParameterGrid,
    exponentialDistribution,
    linearDistribution,
    setDistribution,
)


class BasicSamplerTests(unittest.TestCase):
    def test_grid_density_one_uses_midpoint_for_each_variable(self):
        sampler = basicSampler(x=(0.0, 10.0), y=(2.0, 6.0))

        points = sampler.grid(1)

        np.testing.assert_allclose(points, np.array([[5.0, 4.0]]))
        self.assertEqual(sampler.getPoints(1), [{"x": 5.0, "y": 4.0}])

    def test_get_sample_maps_normalized_value_into_variable_ranges(self):
        sampler = basicSampler(x=(0.0, 10.0), y=(20.0, 30.0))

        result = sampler.getSample([0.25, 0.5])

        np.testing.assert_allclose(result, np.array([2.5, 25.0]))

    def test_sobol_sample_respects_range_bounds_and_seed(self):
        sampler = basicSampler(x=(0.0, 2.0), y=(10.0, 12.0))

        points = sampler.sobol(2, seed=123)

        self.assertEqual(points.shape, (4, 2))
        self.assertTrue(np.all(points[:, 0] >= 0.0))
        self.assertTrue(np.all(points[:, 0] <= 2.0))
        self.assertTrue(np.all(points[:, 1] >= 10.0))
        self.assertTrue(np.all(points[:, 1] <= 12.0))


class DistributionTests(unittest.TestCase):
    def test_set_distribution_iterates_in_defined_order(self):
        values = setDistribution(["A", "B", "C"])

        self.assertEqual(len(values), 3)
        self.assertEqual(list(values), ["A", "B", "C"])

    def test_linear_and_exponential_distributions_match_expected_ranges(self):
        linear = linearDistribution((0.0, 2.0), density=3)
        exponential = exponentialDistribution((1.0, 100.0), density=3, base=10)

        np.testing.assert_allclose(list(linear), [0.0, 1.0, 2.0])
        np.testing.assert_allclose(list(exponential), [1.0, 10.0, 100.0])
        self.assertAlmostEqual(linear(0.5), 1.0)
        self.assertAlmostEqual(exponential(0.5), 10.0)


class DefinedSamplerTests(unittest.TestCase):
    def test_defined_sampler_rejects_unsupported_variable_types(self):
        with self.assertRaisesRegex(TypeError, "not in the allowed class types"):
            definedSampler(x=object())

    def test_defined_sampler_combines_discrete_and_continuous_samples(self):
        continuous = linearDistribution((0.0, 2.0), density=2)
        discrete = setDistribution(["A", "B"])
        sampler = definedSampler(cut=continuous, material=discrete)

        points = sampler.grid(2)

        self.assertEqual(len(points), 4)
        self.assertEqual(len({sample["material"] for sample in points}), 2)
        np.testing.assert_allclose(
            sorted(sample["cut"] for sample in points),
            [0.0, 0.0, 2.0, 2.0],
        )

    def test_distributed_parameter_grid_returns_normalized_compositions(self):
        grid = distributedParameterGrid(parameters=3, subdivisions=3)

        self.assertTrue(grid.size > 0)
        self.assertTrue(np.all(np.isclose(grid.sum(axis=1), 1.0)))


if __name__ == "__main__":
    unittest.main()
