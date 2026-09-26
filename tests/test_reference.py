import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sparse_indexer.reference import map_logical_to_paged, stable_topk  # noqa: E402


class StableTopKTests(unittest.TestCase):
    def test_selects_highest_valid_positions(self):
        scores = [[0.1, 0.9, 0.4, 0.8], [0.7, 0.2, 0.6, 0.5]]
        self.assertEqual(stable_topk(scores, k=2), [[1, 3], [0, 2]])

    def test_masks_tail_and_handles_k_larger_than_valid_length(self):
        scores = [[9.0, 8.0, 100.0], [0.2, 0.2, 0.1]]
        self.assertEqual(
            stable_topk(scores, k=8, valid_lengths=[2, 3]),
            [[0, 1], [0, 1, 2]],
        )

    def test_ties_are_resolved_by_logical_position(self):
        self.assertEqual(stable_topk([[1.0, 2.0, 2.0, 2.0]], k=3), [[1, 2, 3]])

    def test_rejects_non_finite_scores(self):
        with self.assertRaises(ValueError):
            stable_topk([[1.0, float("nan")]], k=1)


class PagedAddressTests(unittest.TestCase):
    def test_maps_across_page_boundaries_per_request(self):
        positions = [[0, 3, 4, 7], [1, 5]]
        page_tables = [[10, 11], [20, 23]]
        mapped = map_logical_to_paged(positions, page_tables, block_size=4)
        self.assertEqual(
            [[(address.page_id, address.offset) for address in row] for row in mapped],
            [[(10, 0), (10, 3), (11, 0), (11, 3)], [(20, 1), (23, 1)]],
        )
        self.assertEqual(mapped[0][2].slot_for(4), 44)

    def test_composes_with_topk_and_respects_valid_length(self):
        scores = [[0.3, 0.9, 0.1, 0.8, 99.0]]
        selected = stable_topk(scores, k=2, valid_lengths=[4])
        mapped = map_logical_to_paged(selected, [[7, 8]], block_size=4, valid_lengths=[4])
        self.assertEqual(selected, [[1, 3]])
        self.assertEqual([(item.page_id, item.offset) for item in mapped[0]], [(7, 1), (7, 3)])

    def test_rejects_unmapped_or_invalid_positions(self):
        with self.assertRaises(ValueError):
            map_logical_to_paged([[4]], [[1]], block_size=4)
        with self.assertRaises(ValueError):
            map_logical_to_paged([[3]], [[1]], block_size=4, valid_lengths=[3])


if __name__ == "__main__":
    unittest.main()
