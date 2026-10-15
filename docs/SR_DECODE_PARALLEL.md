# SR decoding performance patch

vLLM 0.11.0 serializes XGrammar token masks for batches <=128. SR serving uses
max_num_seqs=8 and bounded string grammars, making CPU masks the bottleneck.

scripts/run_sr_parallel_decode.py builds a private import overlay and launches
the existing coordinator. Only the mask parallel threshold (128 -> 1) and
chunk size (16 -> 1) change. XGrammar 0.1.25, model weights, grammar definitions,
sampling settings, prompts, and reflection limits remain the same. Installed
site-packages are untouched. The process environment propagates the overlay to
new SR services; existing services require a controlled restart.

Resume with:

    /home/shn/miniconda3/envs/verl-agent/bin/python scripts/run_sr_parallel_decode.py --output outputs/wm_reflection/ee_sr_agentwm_prompts_v2 --compact-gpus

Before deployment, run the actual StructuredOutputManager serial/parallel paths
with real token masks for ALF review, ScienceWorld critique and ScienceWorld
replan. The test checks every mask at every token for four concurrent matchers:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=outputs/wm_reflection_setup/sr_decode_parallel:. /home/shn/miniconda3/envs/verl-agent/bin/python scripts/test_sr_parallel_masks.py

Measurements and the version/hash manifest are stored under
outputs/wm_reflection_setup/sr_decode_parallel. cutover.json records completed
result hashes, interrupted tasks, and process identities at service migration.
Completed results must remain byte-identical; interrupted tasks are retried by
the existing task queue. This is an inference runtime change, not a fresh
scientific replicate. Batch scheduling may affect numerical reproducibility;
mask equality alone does not prove identical sampled trajectories.

Rollback: stop the recorded coordinator and SR services, preserve completed
results, and resume using scripts/run_sr_experiments.py without the overlay on
PYTHONPATH. Never stop services belonging to other users.
