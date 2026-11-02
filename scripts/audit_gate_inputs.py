#!/usr/bin/env python3
"""Recheck accepted signals and every source case before Phase4 invocation use."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experience.memory import ExperienceMemory
from experience.schema import validate_case
from experience.splits import file_hash,validate_manifest
from gates.core import fingerprint

if __name__=="__main__":
    root=Path(__file__).resolve().parents[1]
    m=json.loads((root/"docs/phase3/split_manifest.json").read_text());validate_manifest(m,require_native=True)
    snapshot=json.loads((root/"docs/phase3/memory_snapshot.json").read_text());memory=ExperienceMemory(snapshot,m)
    for c in memory.cases:validate_case(c,m)
    probes=[c for c in memory.cases if c["provenance"]["collector"].startswith("controlled-real-environment")]
    actor=json.loads((root/"configs/internal_signals/scienceworld_token_nll_v1.json").read_text())
    cfg=json.loads((root/"configs/gates/provisional_full_v1.json").read_text())
    if cfg["internal_signature"]["config_sha256"]!=fingerprint(actor):raise ValueError("Changed internal config")
    subprocess.run(["git","diff","--exit-code","2b6fde0","--","internal_signals","configs/internal_signals","docs/PAPER_ALIGNED_EVAL.md"],cwd=root,check=True,capture_output=True)
    result={"internal_backend":actor["backend"],"model_revision":actor["model_revision"],"dtype":actor["dtype"],
        "internal_code_config_and_paper_protocol_unchanged_from_accepted_phase3":True,
        "cases":len(memory.cases),"all_cases_approved_training":True,"controlled_probe_steps":len(probes),
        "probe_task_ids":sorted({c["task_id"] for c in probes}),"cards":len(memory.cards),
        "snapshot_sha256":file_hash(root/"docs/phase3/memory_snapshot.json"),
        "tri_state_values":["true","false","unknown"],"natural_positive_sufficiency_examples":0,
        "coverage_limit":"two door predicates; no WM utility evidence; semantic/template independence not established"}
    out=root/"outputs/phase4_setup/input_audit.json";out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
