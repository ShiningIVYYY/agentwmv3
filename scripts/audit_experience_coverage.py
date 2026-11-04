#!/usr/bin/env python3
"""Training evidence inventory; no generic rules inferred from ambiguous failures."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experience.memory import ExperienceMemory


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--snapshot",required=True);p.add_argument("--manifest",default="docs/phase3/split_manifest.json");p.add_argument("--output",required=True);a=p.parse_args()
    memory=ExperienceMemory(json.loads(Path(a.snapshot).read_text()),json.loads(Path(a.manifest).read_text()))
    cases=memory.cases
    transitions=defaultdict(list)
    for c in cases:transitions[(c["episode_id"],c["observation"],c["draft_action"],c["environment_feedback"])].append(c["case_id"])
    group=lambda predicate:[c for c in cases if predicate(c)]
    subsets={"navigation":group(lambda c:c["draft_action"].startswith("go ")),
        "object_interaction":group(lambda c:c["draft_action"].startswith(("focus on ","look at ","use ","connect "))),
        "state_preconditions":group(lambda c:"door" in c["environment_feedback"].casefold()),
        "failed_actions":group(lambda c:"no known action" in c["environment_feedback"].casefold() or "not sure how" in c["environment_feedback"].casefold()),
        "positive_native_reward":group(lambda c:c["reward"] is not None and c["reward"]>0)}
    inventory={k:{"cases":len(v),"episodes":len({c["episode_id"] for c in v}),"goal_groups":len({c["goal_group"] for c in v}),"source_case_ids":[c["case_id"] for c in v]} for k,v in subsets.items()}
    report={"training_cases":len(cases),"all_sources_validated_train_only":True,"inventory":inventory,
        "repeated_identical_transitions_within_episode":[ids for ids in transitions.values() if len(ids)>1],
        "cards":[{k:c[k] for k in ["card_id","kind","reliability","independent_episodes","independent_instances","independent_goal_groups","supporting_cases","counterexamples"]} for c in memory.cards],
        "unpromoted_candidates":{"tool_use":"specific failed battery/tool examples lack broad independent conditional evidence",
            "object_interaction":"referent availability, alias and task condition differ; no blanket focus/use correction",
            "failed_actions":"No known action has multiple causes; feedback alone does not identify a universal failure rule",
            "repeated_actions":"within-episode repeats are not independent support and need an explicit history-aware query contract",
            "subgoal_progress":"few observed positive rewards do not justify a general causal subgoal rule",
            "wm_utility":"no trustworthy Call/Skip counterfactual controls in training cases"},
        "new_training_gpu_collection":0,"reliability_threshold_relaxed":False,
        "broader_domain_coverage_still_needed":True}
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"training_cases":len(cases),"cards":len(memory.cards),"inventory":{k:v["cases"] for k,v in inventory.items()}},indent=2))
