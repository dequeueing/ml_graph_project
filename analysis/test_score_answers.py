"""Regression checks for answer extraction and graph-based adjudication."""

import unittest

from score_answers import adjudicate, extract_path, normalized
from audit_node_scores import extract_node_label


def case(task, n, edges, **query):
    return {"task": task, "graph_data": {"num_nodes": n, "edges": edges},
            "query_params": query}


class ScoringTests(unittest.TestCase):
    def test_observed_topological_sort_format(self):
        graph = case("topology", 8, [[0, 7], [0, 4], [2, 6], [2, 4],
                                    [3, 2], [3, 0], [3, 7], [3, 1], [3, 5], [7, 5]])
        answer = "Topological sort: 3, 1, 0, 7, 5, 2, 4, 6."
        self.assertIsNone(adjudicate(graph, answer, legacy=True)[0])
        self.assertTrue(adjudicate(graph, answer)[0])

    def test_missing_isolated_vertex_is_permitted(self):
        graph = case("topology", 7, [[2, 5], [2, 1], [2, 3], [4, 2], [4, 0], [5, 3]])
        self.assertTrue(adjudicate(graph, "Topological sort: 4, 0, 2, 5, 1, 3.")[0])
        self.assertFalse(adjudicate(graph, "Topological sort: 4, 0, 2, 1, 3.")[0])

    def test_invalid_topological_order_is_not_rescued(self):
        graph = case("topology", 3, [[0, 1], [1, 2]])
        self.assertFalse(adjudicate(graph, "Candidate: 0, 1, 2.\nFinal answer: 2, 1, 0.")[0])

    def test_repeated_path_and_weight_suffix(self):
        graph = case("shortest_path", 4, [[0, 1, 1], [1, 3, 1], [0, 2, 1], [2, 3, 8]],
                     source=0, target=3)
        answer = "The minimum weight is 2 via path 0 -> 1 -> 3. The answer is: 0 -> 1 -> 3, and the shortest path length is 2."
        self.assertIsNone(adjudicate(graph, answer, legacy=True)[0])
        self.assertTrue(adjudicate(graph, answer)[0])

    def test_final_wrong_path_overrides_correct_candidate(self):
        graph = case("shortest_path", 4, [[0, 1, 1], [1, 3, 1], [0, 2, 1], [2, 3, 8]],
                     source=0, target=3)
        answer = "Candidate path: 0 -> 1 -> 3.\nThe answer is: 0 -> 2 -> 3, with total weight 9."
        self.assertFalse(adjudicate(graph, answer)[0])

    def test_unlabeled_intermediate_path_is_not_used(self):
        answer = "Candidate paths:\n0 -> 1 -> 3: weight 2\nI cannot determine the answer."
        self.assertIsNone(extract_path(normalized(answer))[0])

    def test_labeled_answer_before_weight_footer(self):
        graph = case("shortest_path", 3, [[0, 1, 2], [1, 2, 4]], source=0, target=2)
        answer = "Final answer: [0, 1, 2].\nThe total weight is 6."
        self.assertTrue(adjudicate(graph, answer)[0])

    def test_unique_arrow_path_with_unicode_and_markdown(self):
        answer = "The path is **0 → 1 → 2**, with total weight 6."
        self.assertEqual(extract_path(normalized(answer))[0], (0, 1, 2))

    def test_duplicate_vertex_is_invalid(self):
        graph = case("hamilton", 3, [[0, 1], [1, 2]])
        self.assertFalse(adjudicate(graph, "Answer: 0, 1, 0, 2.")[0])

    def test_negative_hamiltonian_answers(self):
        chain = case("hamilton", 3, [[0, 1], [1, 2]])
        star = case("hamilton", 4, [[0, 1], [0, 2], [0, 3]])
        self.assertFalse(adjudicate(chain, "No.")[0])
        self.assertTrue(adjudicate(star, "No.")[0])

    def test_final_binary_label_overrides_earlier_label(self):
        graph = case("cycle", 3, [[0, 1], [1, 2], [2, 0]])
        self.assertFalse(adjudicate(graph, "Yes.\nThe answer is No.")[0])

    def test_explanatory_edge_list_does_not_override_answer(self):
        graph = case("topology", 3, [[0, 1], [1, 2]])
        text = "A valid topological ordering is:\n0, 1, 2\nAll edges point forward in this order: 0 -> 1, 1 -> 2."
        self.assertTrue(adjudicate(graph, text)[0])

    def test_reasoning_fragment_does_not_override_declared_order(self):
        text = "A valid topological ordering is:\n0, 1, 2, 3\nThe initial zero-indegree nodes are:\n0, 1\nRepeated removal gives the ordering above."
        self.assertEqual(extract_path(text)[0], (0, 1, 2, 3))

    def test_corrected_direction_overrides_bare_path(self):
        text = "0 -> 1 -> 2 (reversed), i.e., the shortest path from node 2 to node 0 is 2 -> 1 -> 0, and the shortest path length is 3."
        self.assertEqual(extract_path(text)[0], (2, 1, 0))

    def test_latex_answer_before_reasoning(self):
        text = r"A valid topological ordering is:" + "\n" + r"\boxed{0,\ 1,\ 2}" + "\nAll edges are respected."
        self.assertEqual(extract_path(text)[0], (0, 1, 2))

    def test_direct_edge_question_is_not_final_binary_answer(self):
        graph = case("connectivity", 3, [[0, 1], [1, 2]], source=0, target=2)
        text = "Yes. Is there a direct edge? No, but 0 -> 1 -> 2 is a path."
        self.assertTrue(adjudicate(graph, text)[0])

    def test_alternative_path_does_not_override_final_answer(self):
        text = "Shortest path: 0 -> 1, total weight 2.\nThe alternative route 0 -> 2 -> 1 costs 9."
        self.assertEqual(extract_path(text)[0], (0, 1))

    def test_distance_footer_is_not_a_one_vertex_path(self):
        text = "Shortest path: 0 -> 1 -> 2.\nAnswer: 6"
        self.assertEqual(extract_path(text)[0], (0, 1, 2))

    def test_tool_call_is_not_a_final_answer(self):
        text = '<DSML>print("Answer: 0 -> 1 -> 2")</DSML>'
        self.assertIsNone(extract_path(text)[0])

    def test_final_impossibility_overrides_rejected_partial_path(self):
        graph = case("hamilton", 4, [[0, 1], [0, 2], [0, 3]])
        text = "Yes. One path is 1 -> 0 -> 2. This omits node 3, so no Hamiltonian path exists."
        self.assertTrue(adjudicate(graph, text)[0])

    def test_failure_to_provide_path_is_not_proof_of_impossibility(self):
        graph = case("hamilton", 4, [[0, 1], [0, 2], [0, 3]])
        text = "Yes. One path is 1 -> 0 -> 2. This is incomplete, so no Hamiltonian path is provided."
        self.assertFalse(adjudicate(graph, text)[0])

    def test_undirected_path_reversal_is_equivalent(self):
        graph = case("shortest_path", 3, [[0, 1, 2], [1, 2, 3]], source=2, target=0)
        self.assertTrue(adjudicate(graph, "Shortest path: 0 -> 1 -> 2.")[0])

    def test_weighted_edge_tuple_is_not_three_vertices(self):
        graph = case("shortest_path", 3, [[0, 2, 1]], source=0, target=2)
        self.assertTrue(adjudicate(graph, "The shortest path is the direct edge (0, 2, 1). Total weight 1.")[0])

    def test_coverage_set_does_not_replace_path(self):
        text = "One valid path is 2 -> 0 -> 1.\nThe path visits every node exactly once:\n{0, 1, 2}"
        self.assertEqual(extract_path(text)[0], (2, 0, 1))

    def test_target_prediction_not_neighbors_label(self):
        self.assertEqual(extract_node_label("Node 10612 is label 0. Its neighbor 8656 has label 1.", 10612)[0], 0)

    def test_neighbor_label_does_not_turn_abstention_into_prediction(self):
        self.assertIsNone(extract_node_label("Node 10612 has an unknown label. Its neighbor has label 0, but that does not determine its label.", 10612)[0])

    def test_cautious_prediction_is_still_a_prediction(self):
        self.assertEqual(extract_node_label("Its label cannot be uniquely determined. Its neighbors suggest label 1 if using majority voting.", 42)[0], 1)

    def test_node_tool_call_has_no_prediction(self):
        self.assertIsNone(extract_node_label('<DSML>print("label 1")</DSML>', 42)[0])


if __name__ == "__main__":
    unittest.main()
