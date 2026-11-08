# BiGDN CPU toy demonstration

This isolated demo exercises the repository's teacher-to-student workflow on
one small existing graph. It does not modify the normal training scripts, graph
files, or checkpoints.

## What it uses

- `BIGDN\train_graphs\graph1.txt` only: 26 nodes (IDs `0`–`25`) and 55
  directed edges in the inspected checkout. Its original real-world provenance
  is not specified in the repository README; this demo does not guess it.
- `BIGDN\q_net.pth` as the frozen teacher checkpoint. The demo does not retrain
  or load the node encoder separately.
- The repository's `QValueNet`, `StudentAgent`, replay buffer, student update
  (TD plus teacher KL-distillation loss), and n-step replay normalization.
- Directed edge activation probability `1 / in_degree(target)`, as in
  `main_s.py`.
- CPU execution, seed budget `k=3`, exactly 10 experience episodes and exactly
  10 student optimizer updates.
- Evaluation after updates 1, 5, and 10 using teacher, student, and random
  policies.
- Exactly 100 single-process Independent Cascade Monte Carlo trials for every
  uncached reward or evaluation spread estimate. This is 100 cascade trials
  per estimate, not 100 episodes or student updates. Direct simulation avoids
  multiprocessing overhead while retaining the edge probability and spread
  semantics.

The same graph is used for learning and demonstration evaluation; therefore,
evaluation is non-independent. Results are only for understanding the workflow
and are not comparable to full-dataset results. Fixed Python, NumPy, and
PyTorch seeds improve repeatability, but exact results may depend on the
PyTorch/PyG/torch-scatter and platform environment.

## Dependencies and run command (Windows PowerShell)

The BIGDN checkout's Python 3.10 virtual environment already contains the
versions listed in `requirements.txt`. To install or verify the demo's pinned
dependencies in that same environment:

```powershell
$python = "C:\DISK-D\SEM 7\MAJOR PROJECT\our project\BIGDN\.venv\Scripts\python.exe"
& $python -m pip install -r "C:\DISK-D\SEM 7\MAJOR PROJECT\our project\Base_project\toy_demo\requirements.txt"
```

The runner looks for the BIGDN checkout next to `Base_project` by default. If
you place this demo elsewhere (such as in another GitHub repository), set
`BIGDN_PROJECT_ROOT` to the checkout containing `main_s.py`, `train_graphs`,
and `q_net.pth`:

```powershell
$env:BIGDN_PROJECT_ROOT = "C:\DISK-D\SEM 7\MAJOR PROJECT\our project\BIGDN"
```

Run the demo manually:

```powershell
& $python "C:\DISK-D\SEM 7\MAJOR PROJECT\our project\Base_project\toy_demo\run_demo.py"
```

Run the focused tests:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
& $python -m unittest discover -s "C:\DISK-D\SEM 7\MAJOR PROJECT\our project\Base_project\toy_demo\tests" -v
```

## Workflow

```text
graph1.txt ──> directed graph + 1/in-degree edge probabilities
    ──> student chooses 3 seeds sequentially
    ──> 100-trial IC spread estimates for seed prefixes
    ──> marginal spread rewards / node count ──> replay buffer
    ──> frozen q_net.pth teacher + student TD/KL update
    ──> repeat for exactly 10 episodes and 10 optimizer updates
    ──> same-graph teacher/student/random evaluation ──> saved artifacts
```

For example, after selecting seed `s` from an initially empty seed set, the
demo estimates the expected influenced-node count for `{s}` using 100 IC
realizations. The first marginal reward is that estimate minus the empty-set
spread of zero. For this 26-node graph, the n-step replay reward stored for
learning is the marginal reward divided by 26. Later seed selections use the
change in estimated spread from the preceding seed prefix. The exact first
seed and estimate from a particular run are recorded in its generated report.

## Outputs

Each invocation creates a new timestamped folder in `outputs\` to avoid
overwriting prior results:

- `student_final.pth` — final student state dictionary, separate from the
  repository checkpoints.
- `run_log.json` — seeds, graph/configuration, every training action, periodic
  evaluation, runtime, and reproducibility note.
- `training_steps.csv` — one row per selected training seed, including spread,
  marginal and normalized replay reward, update number, and replay size.
- `run_report.md` — generated run summary and a worked example using actual
  first-step values.

Generated files live under `Base_project\toy_demo\outputs\`. To clean up, first
inspect the timestamped run folders you want to remove, then delete only those
specific folders in PowerShell; the demo itself never deletes outputs.
