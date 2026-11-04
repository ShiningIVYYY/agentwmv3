"""Official task/turn seed namespace; reviews have a separate subcall namespace."""
import hashlib
import json


def request_seed(task_id, turn, rollout_seed=0, *, role="draft", subcall=0, task_seed=0):
    if role == "draft":
        payload = ["eval_request_seed_v1", "scienceworld", task_seed, rollout_seed, task_id, turn]
    else:
        payload = ["agentwmv3_review_seed_v1", "scienceworld", task_seed, rollout_seed, task_id, turn, subcall]
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)
