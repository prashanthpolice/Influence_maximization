import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True

import networkx as nx
import numpy as np
import torch


DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.environ.get("BIGDN_PROJECT_ROOT", DEMO_DIR.parents[1] / "BIGDN")
).resolve()
OUTPUT_ROOT = DEMO_DIR / "outputs"
GRAPH_PATH = PROJECT_ROOT / "train_graphs" / "graph1.txt"
TEACHER_PATH = PROJECT_ROOT / "q_net.pth"

SEED = 20261108
MC_TRIALS = 100
K = 3
STUDENT_UPDATES = 10
EVALUATION_UPDATES = {1, 5, STUDENT_UPDATES}
NUM_FEATURES = 64
BATCH_SIZE = 16
TARGET_UPDATE = 100
GAMMA = 0.99
LEARNING_RATE = 0.001

sys.path.insert(0, str(PROJECT_ROOT))

import utils  # noqa: E402
from agent import StudentAgent, get_q_net_input  # noqa: E402
from environment import GraphEnvironment  # noqa: E402
from models import QValueNet  # noqa: E402


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_toy_graph(path=GRAPH_PATH):
    graph = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph)
    expected_nodes = set(range(graph.number_of_nodes()))
    if set(graph.nodes()) != expected_nodes:
        raise ValueError("Toy graph node IDs must be contiguous integers starting at 0.")
    if not graph.number_of_nodes() or not graph.number_of_edges():
        raise ValueError("Toy graph must contain at least one node and one edge.")

    for source, target in graph.edges():
        graph.edges[source, target]["weight"] = 1.0 / graph.in_degree(target)
    return graph


def estimate_spread(graph, seeds, trials=MC_TRIALS, rng=None):
    if trials <= 0:
        raise ValueError("Monte Carlo trial count must be positive.")
    if any(seed not in graph for seed in seeds):
        raise ValueError("Every seed must be a node in the graph.")

    draw = random.random if rng is None else rng.random
    total_influenced = 0
    for _ in range(trials):
        active = set(seeds)
        frontier = set(seeds)
        while frontier:
            newly_active = set()
            for source in frontier:
                for target in graph.neighbors(source):
                    if target not in active and draw() <= graph.edges[source, target]["weight"]:
                        newly_active.add(target)
            active.update(newly_active)
            frontier = newly_active
        total_influenced += len(active)
    return total_influenced / trials


class MonteCarloEstimator:
    def __init__(self):
        self.calls = 0

    def __call__(self, graph, seeds):
        self.calls += 1
        return estimate_spread(graph, seeds, MC_TRIALS)


class ToyGraphEnvironment(GraphEnvironment):
    def __init__(self, graph, estimator):
        super().__init__([graph], K, gamma=GAMMA, n_steps=1, R=MC_TRIALS, num_workers=1)
        self.estimator = estimator

    def compute_reward(self):
        current_spread = self.estimator(self.graph, self.seeds)
        marginal_reward = current_spread - self.preview_reward
        self.preview_reward = current_spread
        return marginal_reward


def select_greedy_seeds(model, graph, k, device):
    model.eval()
    data = get_q_net_input([graph], NUM_FEATURES, device)
    states = torch.zeros(graph.number_of_nodes(), dtype=torch.float, device=device)
    seeds = []
    with torch.no_grad():
        for _ in range(k):
            q_values = model(data.x, data.edge_index, data.edge_weight, data.batch, states)
            q_values = q_values.masked_fill(states.bool(), float("-inf"))
            seed = int(q_values.argmax().item())
            seeds.append(seed)
            states[seed] = 1.0
    return seeds


def train_student(agent, env, replay_buffer, on_update=None):
    training_records = []
    for update_number in range(1, STUDENT_UPDATES + 1):
        epsilon = max(0.05, 1.0 - (update_number - 1) * 0.95 / (STUDENT_UPDATES - 1))
        agent.epsilon = epsilon
        state = env.reset()
        done = False
        step_records = []

        while not done:
            action = agent.take_action(state, env)
            reward, next_state, done = env.step(action)
            step_records.append(
                {
                    "update": update_number,
                    "selected_seed": int(action),
                    "seed_prefix": [int(seed) for seed in env.seeds],
                    "estimated_spread": float(env.preview_reward),
                    "marginal_reward": float(reward),
                    "normalized_replay_reward": float(reward / env.graph.number_of_nodes()),
                    "mc_trials": MC_TRIALS,
                }
            )
            state = next_state

        env.n_step_add_buffer(replay_buffer)
        states, actions, rewards, next_states, dones, graphs = replay_buffer.sample(BATCH_SIZE)
        agent.update(states, actions, rewards, next_states, graphs, dones)

        replay_size = len(replay_buffer)
        for record in step_records:
            record["replay_buffer_size"] = replay_size
        training_records.extend(step_records)

        final_spread = step_records[-1]["estimated_spread"]
        selected_seeds = [record["selected_seed"] for record in step_records]
        print(
            f"Student update {update_number}/{STUDENT_UPDATES}: "
            f"seeds={selected_seeds}, estimated spread={final_spread:.3f}, "
            f"final marginal reward={step_records[-1]['marginal_reward']:.3f}, "
            f"replay buffer={replay_size}, MC trials/uncached estimate={MC_TRIALS}"
        )
        if on_update is not None:
            on_update(update_number)

    if agent.count != STUDENT_UPDATES:
        raise RuntimeError(f"Expected {STUDENT_UPDATES} student updates, got {agent.count}.")
    return training_records


def evaluate(model, graph, name, estimator, rng=None, update_number=None):
    seeds = select_greedy_seeds(model, graph, K, torch.device("cpu")) if model else (
        rng.sample(list(graph.nodes()), K)
    )
    spread = estimator(graph, seeds)
    return {
        "policy": name,
        "seeds": seeds,
        "estimated_spread": spread,
        "mc_trials": MC_TRIALS,
        "evaluation_is_independent": False,
        "after_student_update": update_number,
    }


def write_run_outputs(run_dir, run_log):
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "student_final.pth"
    torch.save(run_log.pop("_student_state_dict"), checkpoint_path)
    run_log["checkpoint"] = str(checkpoint_path)

    log_path = run_dir / "run_log.json"
    log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")

    csv_path = run_dir / "training_steps.csv"
    fields = [
        "update",
        "selected_seed",
        "seed_prefix",
        "estimated_spread",
        "marginal_reward",
        "normalized_replay_reward",
        "replay_buffer_size",
        "mc_trials",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in run_log["training_steps"]:
            row = record.copy()
            row["seed_prefix"] = json.dumps(row["seed_prefix"])
            writer.writerow(row)

    first_step = run_log["training_steps"][0]
    lines = [
        "# BiGDN toy demo run report",
        "",
        f"- Run ID: `{run_log['run_id']}`",
        f"- Runtime: {run_log['elapsed_seconds']:.2f} seconds on {run_log['device']}",
        f"- Graph: `{run_log['configuration']['graph_path']}` "
        f"({run_log['configuration']['nodes']} nodes, "
        f"{run_log['configuration']['directed_edges']} directed edges)",
        f"- Training: {run_log['configuration']['episodes']} episodes; "
        f"{run_log['configuration']['student_optimizer_updates']} student updates",
        f"- Seed budget: {run_log['configuration']['seed_budget']}",
        f"- Monte Carlo: exactly "
        f"{run_log['configuration']['monte_carlo_trials_per_uncached_estimate']} "
        "IC trials per uncached estimate "
        f"({run_log['configuration']['uncached_spread_estimate_calls']} estimates; "
        f"{run_log['configuration']['total_ic_cascade_trials']} cascade trials total)",
        "",
        "## Workflow demonstrated",
        "",
        "The runner loads one directed graph, assigns each edge probability "
        "`1 / in_degree(target)`, loads `q_net.pth` as a frozen teacher, and "
        "initializes the repository's student agent. In each episode the student "
        "selects three nodes sequentially. The direct single-process IC simulator "
        "estimates spread for each seed prefix using 100 trials; reward is the "
        "marginal change in expected influenced-node count. Existing n-step replay "
        "logic normalizes that reward by the node count. The student then receives "
        "one TD/KL update from sampled replay experiences. After updates 1, 5, and "
        "10, teacher, student, and random selections are evaluated on the same graph.",
        "",
        "```text",
        "graph -> edge probabilities -> seed actions -> IC spread/reward",
        "  -> replay buffer -> frozen teacher + student update (x10)",
        "  -> non-independent same-graph evaluation -> checkpoint and logs",
        "```",
        "",
        "## Worked first-action example",
        "",
        f"On update 1, the first selected seed was node "
        f"`{first_step['selected_seed']}`. Its estimated spread for that one-node "
        f"seed prefix was **{first_step['estimated_spread']:.3f} influenced nodes** "
        f"over {first_step['mc_trials']} IC trials. The initial empty seed set has "
        "spread zero, so the first marginal reward was "
        f"**{first_step['marginal_reward']:.3f}**. The replay reward stored for "
        f"learning was that marginal reward divided by {run_log['configuration']['nodes']}, "
        f"or **{first_step['normalized_replay_reward']:.6f}**. These are distinct "
        "quantities: the spread is an expected node count, while the normalized "
        "value is the learning reward.",
        "",
        "## Demonstration evaluation",
        "",
        "The same graph is used for training and evaluation, so this is "
        "**non-independent evaluation**, not a generalization result.",
        "",
        "| After update | Policy | Selected seeds | Estimated spread | IC trials |",
        "|---:|---|---:|---:|---:|",
    ]
    for result in run_log["evaluations"]:
        lines.append(
            f"| {result['after_student_update']} | {result['policy']} | {result['seeds']} | "
            f"{result['estimated_spread']:.3f} | {result['mc_trials']} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility and artifacts",
            "",
            run_log["reproducibility_note"],
            "",
            f"- Student checkpoint: `{checkpoint_path.name}`",
            f"- Machine-readable run log: `{log_path.name}`",
            f"- Training-step CSV: `{csv_path.name}`",
        ]
    )
    (run_dir / "run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checkpoint_path, log_path, csv_path


def main():
    started_at = time.perf_counter()
    seed_everything()
    torch.set_num_threads(1)
    device = torch.device("cpu")

    if not GRAPH_PATH.is_file():
        raise FileNotFoundError(f"Expected repository graph was not found: {GRAPH_PATH}")
    if not TEACHER_PATH.is_file():
        raise FileNotFoundError(f"Teacher checkpoint was not found: {TEACHER_PATH}")

    graph = load_toy_graph()
    estimator = MonteCarloEstimator()
    env = ToyGraphEnvironment(graph, estimator)
    replay_buffer = utils.ReplayBuffer(capacity=500)

    teacher = QValueNet(NUM_FEATURES).to(device)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device, weights_only=True))
    teacher.eval()
    teacher.requires_grad_(False)

    agent = StudentAgent(
        NUM_FEATURES,
        GAMMA,
        epsilon=1.0,
        lr=LEARNING_RATE,
        device=device,
        teacher=teacher,
        target_update=TARGET_UPDATE,
        n_steps=1,
    )
    evaluations = []

    def evaluate_checkpoint(update_number):
        if update_number not in EVALUATION_UPDATES:
            return
        checkpoint_results = [
            evaluate(teacher, graph, "teacher", estimator, update_number=update_number),
            evaluate(agent.q_net, graph, "student", estimator, update_number=update_number),
            evaluate(
                None,
                graph,
                "random",
                estimator,
                rng=random.Random(SEED + update_number),
                update_number=update_number,
            ),
        ]
        evaluations.extend(checkpoint_results)
        print(
            f"Non-independent evaluation after update {update_number}: "
            f"{[(item['policy'], item['seeds'], item['estimated_spread']) for item in checkpoint_results]}"
        )

    training_steps = train_student(agent, env, replay_buffer, evaluate_checkpoint)
    expected_estimate_calls = STUDENT_UPDATES * K + len(EVALUATION_UPDATES) * 3
    if estimator.calls != expected_estimate_calls:
        raise RuntimeError(
            f"Expected {expected_estimate_calls} uncached estimates, got {estimator.calls}."
        )

    elapsed_seconds = time.perf_counter() - started_at
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = OUTPUT_ROOT / f"run_{run_id}"
    run_log = {
        "run_id": run_id,
        "device": str(device),
        "seeds": {
            "python": SEED,
            "numpy": SEED,
            "pytorch": SEED,
            "random_baseline": SEED + 1,
            "cuda_seeded_if_available": torch.cuda.is_available(),
        },
        "configuration": {
            "graph_path": str(GRAPH_PATH),
            "graph_role": "single small repository training graph; evaluation reuses it",
            "nodes": graph.number_of_nodes(),
            "directed_edges": graph.number_of_edges(),
            "edge_probability": "1 / in_degree(target)",
            "seed_budget": K,
            "episodes": STUDENT_UPDATES,
            "student_optimizer_updates": agent.count,
            "monte_carlo_trials_per_uncached_estimate": MC_TRIALS,
            "uncached_spread_estimate_calls": estimator.calls,
            "total_ic_cascade_trials": estimator.calls * MC_TRIALS,
            "spread_estimate_cache_enabled": False,
            "reward": "marginal change in expected influenced-node count",
            "replay_reward_scale": "marginal reward / number of nodes (n_steps=1)",
            "target_network_update_interval": TARGET_UPDATE,
            "teacher_checkpoint": str(TEACHER_PATH),
        },
        "training_steps": training_steps,
        "evaluations": evaluations,
        "evaluation_is_independent": False,
        "elapsed_seconds": elapsed_seconds,
        "reproducibility_note": (
            "Seeds are fixed and CPU execution is selected. Exact reproducibility can "
            "still depend on PyTorch, PyG, torch-scatter, and platform versions."
        ),
        "_student_state_dict": agent.q_net.state_dict(),
    }
    checkpoint_path, log_path, csv_path = write_run_outputs(run_dir, run_log)
    print(f"Completed exactly {agent.count} student updates in {elapsed_seconds:.2f}s.")
    print(f"Evaluation (non-independent, same graph): {evaluations}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"JSON run log: {log_path}")
    print(f"Training CSV: {csv_path}")
    print(f"Run report: {run_dir / 'run_report.md'}")


if __name__ == "__main__":
    main()
