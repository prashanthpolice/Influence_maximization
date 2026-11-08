# BiGDN toy demo run report

- Run ID: `20261108_172019_021908`
- Runtime: 1.07 seconds on cpu
- Graph: `C:\DISK-D\SEM 7\MAJOR PROJECT\our project\BIGDN\train_graphs\graph1.txt` (26 nodes, 55 directed edges)
- Training: 10 episodes; 10 student updates
- Seed budget: 3
- Monte Carlo: exactly 100 IC trials per uncached estimate (39 estimates; 3900 cascade trials total)

## Workflow demonstrated

The runner loads one directed graph, assigns each edge probability `1 / in_degree(target)`, loads `q_net.pth` as a frozen teacher, and initializes the repository's student agent. In each episode the student selects three nodes sequentially. The direct single-process IC simulator estimates spread for each seed prefix using 100 trials; reward is the marginal change in expected influenced-node count. Existing n-step replay logic normalizes that reward by the node count. The student then receives one TD/KL update from sampled replay experiences. After updates 1, 5, and 10, teacher, student, and random selections are evaluated on the same graph.

```text
graph -> edge probabilities -> seed actions -> IC spread/reward
  -> replay buffer -> frozen teacher + student update (x10)
  -> non-independent same-graph evaluation -> checkpoint and logs
```

## Worked first-action example

On update 1, the first selected seed was node `7`. Its estimated spread for that one-node seed prefix was **2.790 influenced nodes** over 100 IC trials. The initial empty seed set has spread zero, so the first marginal reward was **2.790**. The replay reward stored for learning was that marginal reward divided by 26, or **0.107308**. These are distinct quantities: the spread is an expected node count, while the normalized value is the learning reward.

## Demonstration evaluation

The same graph is used for training and evaluation, so this is **non-independent evaluation**, not a generalization result.

| After update | Policy | Selected seeds | Estimated spread | IC trials |
|---:|---|---:|---:|---:|
| 1 | teacher | [0, 5, 15] | 21.660 | 100 |
| 1 | student | [2, 15, 0] | 17.160 | 100 |
| 1 | random | [5, 1, 15] | 14.120 | 100 |
| 5 | teacher | [0, 5, 15] | 21.510 | 100 |
| 5 | student | [2, 15, 0] | 17.230 | 100 |
| 5 | random | [5, 0, 4] | 17.740 | 100 |
| 10 | teacher | [0, 5, 15] | 22.150 | 100 |
| 10 | student | [2, 15, 0] | 16.600 | 100 |
| 10 | random | [20, 6, 10] | 6.450 | 100 |

## Reproducibility and artifacts

Seeds are fixed and CPU execution is selected. Exact reproducibility can still depend on PyTorch, PyG, torch-scatter, and platform versions.

- Student checkpoint: `student_final.pth`
- Machine-readable run log: `run_log.json`
- Training-step CSV: `training_steps.csv`
