import math
import unittest

import numpy as np

from heteroweave import (
    clas_from_activations,
    clas_from_pattern_counts,
    nondominated_indices,
    validate_composition,
)


class CLASTest(unittest.TestCase):
    def test_square_root_aggregation(self):
        self.assertAlmostEqual(clas_from_pattern_counts([4, 9]), 5.0)

    def test_activation_patterns(self):
        activations = [
            np.asarray([[1.0, -1.0, 2.0], [-1.0, 2.0, 3.0]]),
            np.asarray([[1.0, 1.0], [1.0, -1.0]]),
        ]
        self.assertAlmostEqual(
            clas_from_activations(activations), math.sqrt(3) + math.sqrt(2)
        )


class CompositionTest(unittest.TestCase):
    def test_valid_variable_cardinality(self):
        result = validate_composition([("a",), ("b", "c")], [2, 2])
        self.assertEqual(result, (("a",), ("b", "c")))

    def test_duplicate_identifier_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_composition([("mae", "mae")], [2])


class ParetoTest(unittest.TestCase):
    def test_three_objective_front(self):
        points = [
            (10.0, 4.0, 4.0),
            (9.0, 3.0, 3.0),
            (8.0, 5.0, 5.0),
            (10.0, 5.0, 4.0),
        ]
        self.assertEqual(nondominated_indices(points), [0, 1])


if __name__ == "__main__":
    unittest.main()
