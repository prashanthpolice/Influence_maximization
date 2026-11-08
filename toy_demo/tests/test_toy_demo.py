import random
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_demo
from utils import ReplayBuffer


class ToyDemoTests(unittest.TestCase):
    def test_selected_repository_graph_shape_and_edge_weights(self):
        graph = run_demo.load_toy_graph()
        self.assertEqual(graph.number_of_nodes(), 26)
        self.assertEqual(graph.number_of_edges(), 55)
        self.assertEqual(set(graph.nodes()), set(range(26)))
        for _, target, data in graph.edges(data=True):
            self.assertAlmostEqual(data["weight"], 1 / graph.in_degree(target))

    def test_ic_cascade_with_certain_edges(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(3))
        graph.add_edge(0, 1, weight=1.0)
        graph.add_edge(1, 2, weight=1.0)
        self.assertEqual(
            run_demo.estimate_spread(graph, [0], trials=run_demo.MC_TRIALS, rng=random.Random(3)),
            3.0,
        )

    def test_reward_is_marginal_spread_and_replay_reward_is_normalized(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(3))
        graph.add_edge(0, 1, weight=1.0)
        graph.add_edge(1, 2, weight=1.0)
        graph.add_edge(2, 0, weight=1.0)
        env = run_demo.ToyGraphEnvironment(graph, lambda _, seeds: float(len(seeds)))
        replay = ReplayBuffer(20)
        env.reset()
        env.step(0)
        env.step(1)
        env.step(2)
        env.n_step_add_buffer(replay)
        self.assertEqual([item[2] for item in replay.buffer], [1 / 3, 1 / 3, 1 / 3])

    def test_every_uncached_estimate_uses_exactly_100_trials(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(2))
        graph.add_edge(0, 1, weight=1.0)
        original_estimator = run_demo.estimate_spread
        calls = []

        def counting_estimator(current_graph, seeds, trials=run_demo.MC_TRIALS, rng=None):
            calls.append(trials)
            return original_estimator(current_graph, seeds, trials=trials, rng=rng)

        env = run_demo.ToyGraphEnvironment(graph, counting_estimator)
        env.reset()
        env.step(0)
        env.step(1)
        self.assertEqual(calls, [100, 100])
        self.assertEqual(run_demo.MC_TRIALS, 100)

    def test_training_loop_performs_exactly_ten_updates(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(4))
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        nx.set_edge_attributes(graph, 1.0, "weight")
        env = run_demo.ToyGraphEnvironment(graph, lambda _, seeds: float(len(seeds)))

        class FakeAgent:
            epsilon = 1.0

            def __init__(self):
                self.count = 0

            def take_action(self, state, current_env):
                return next(node for node in range(len(state)) if node not in current_env.seeds)

            def update(self, *args):
                self.count += 1

        agent = FakeAgent()
        buffer = ReplayBuffer(100)
        records = run_demo.train_student(agent, env, buffer)
        self.assertEqual(agent.count, 10)
        self.assertEqual(len(records), 10 * run_demo.K)
        self.assertEqual(len(buffer), 10 * run_demo.K)


if __name__ == "__main__":
    unittest.main()
