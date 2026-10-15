# ScienceWorld SR inference scaling

The existing SR experiment continues in
outputs/wm_reflection/ee_sr_agentwm_prompts_v2.

ScienceWorld has one shared local HTTP router at port 8292. It forwards each
request, with its body unchanged, to the healthy replica with the fewest active
requests; call count breaks ties. Health checks run every five seconds.
The router never automatically repeats a generation request. GET /router_status
reports per-replica active requests, calls, errors and health.

Current layout:

| GPU | Service | Port |
|---|---|---|
| 0 | ALFWorld SR | 8201 |
| 1 | ALFWorld Word2World | 8111 |
| 2 | ScienceWorld Word2World | 8112 |
| 3 | ScienceWorld SR | 8232 |
| 4 | ScienceWorld SR | 8242 |
| 5 | ScienceWorld SR | 8252 |
| 6 | ScienceWorld SR | 8202 |
| 7 | ScienceWorld SR | 8272 |

Oracle and Word2World ScienceWorld groups each have 20 independent environment
workers (40 total). ALFWorld retains four workers per group (8 total).
Every model replica retains max_num_seqs=8 and the private small-batch grammar
mask overlay. The two WebShop services on GPUs 3/5 were stopped only after all
three WebShop arms completed. Resume no longer requires WM services for an
environment whose three full jobs have all completed.

The coordinator accepts optional per-job worker counts and preserves original
preflight endpoint configurations. The evaluation CLI accepts 1..32 workers.
All workers within a group use one endpoint and the existing exclusive task
locks; queue configurations record the expanded worker count.

Model weights, prompts, token limits, horizons, reflection rounds, task manifests,
environment seeds and sampling settings are retained. Different scheduling and
batching may affect sampled trajectories, even with identical request seeds.

Transition records are under outputs/wm_reflection_setup/sr_science_scale:
schedule_transition.json records 4->12 workers per arm, worker_limit_repair.json
records the CLI limit correction, and stage2/transition.json records 12->20.
Each transition archives previous configurations and completed result hashes.
Only interrupted tasks are retried; completed task JSON files are retained.
Intentional maintenance restarts and invalid-CLI startup attempts are recorded
separately so they do not consume the ordinary runner failure retry budget.

Resume:

    /home/shn/miniconda3/envs/verl-agent/bin/python scripts/run_sr_parallel_decode.py --output outputs/wm_reflection/ee_sr_agentwm_prompts_v2 --compact-gpus

The frozen experiment manifest and its runstate hash include each scheduler
transition and the serving router source. Live progress measurements are stored
separately so they do not alter frozen inputs.

Checks cover concurrent forwarding with exact HTTP body preservation, balanced
replica assignment and failed-replica exclusion, full-job versus preflight worker
counts, retained benchmark controls, and expanded worker CLI parsing.
