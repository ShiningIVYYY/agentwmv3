#!/usr/bin/env python3
"""Published expert-corpus overlap audit; no model training, environment or test rollout."""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experience.splits import file_hash,goal_key
from evaluation.independent_dev import atomic


def sequence_hash(texts):return hashlib.sha256(json.dumps(texts,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def task_goal(message):return message.split('\nThis room is called ',1)[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--agenttraj-source',required=True);a=ap.parse_args()
    expert=ROOT/'outputs/ee_sr_audit/expert_sft.jsonl';source=Path(a.agenttraj_source)
    manifest=json.loads((ROOT/'docs/phase3/split_manifest.json').read_text())
    assert file_hash(source)==manifest['source_hashes']['goal_metadata']
    originals=json.loads(source.read_text());index=defaultdict(list)
    for x in originals:
        obs=[m['value'] for m in x['conversations'][2:] if m['from']=='human']
        index[sequence_hash(obs)].append(x['item_id'])
    policy_index=defaultdict(list)
    def policy_key(goal,actions):return sequence_hash([goal_key(task_goal(goal)),actions])
    for x in originals:
        c=x['conversations'];actions=[m['value'].rsplit('Action:',1)[-1].strip() for m in c[3:] if m['from']=='gpt']
        policy_index[policy_key(c[2]['value'],actions)].append(x['item_id'])
    rows=[json.loads(x) for x in expert.read_text().splitlines()];policy_matches=[]
    for i,x in enumerate(rows):
        m=x['messages'];actions=[v['content'].rsplit('Action:',1)[-1].strip() for v in m[3:] if v['role']=='assistant']
        candidates=policy_index.get(policy_key(m[2]['content'],actions),[])
        if candidates:policy_matches.append(dict(expert_row=i,reference_candidate_ids=candidates,unique_within_reference_corpus=len(candidates)==1))
    policy_ids={m['reference_candidate_ids'][0] for m in policy_matches if m['unique_within_reference_corpus']}
    groups=defaultdict(list);unique_matches=[];ambiguous=[];missing=[]
    for i,row in enumerate(rows):
        messages=row['messages'];goal=task_goal(messages[2]['content']);groups[goal_key(goal)].append(i)
        users=[m['content'] for m in messages[2:] if m['role']=='user'];matched=index.get(sequence_hash(users),[])
        if len(matched)==1:unique_matches.append(dict(expert_row=i,task_id=matched[0]))
        elif matched:ambiguous.append(dict(expert_row=i,candidate_ids=matched))
        else:missing.append(i)
    matched_ids={x['task_id'] for x in unique_matches};partitions={}
    for split,items in manifest['partitions'].items():
        partitions[split]=dict(instances=len(items),unique_goal_and_expert_action_sequence_reference_matches=[r['task_id'] for r in items if r['task_id'] in policy_ids],unique_exact_dialogue_id_matches=[r['task_id'] for r in items if r['task_id'] in matched_ids],
            normalized_goal_overlap_ids=[r['task_id'] for r in items if r['goal_group'] in groups],
            unique_goals=len({r['goal_group'] for r in items}),overlapping_goal_groups=sorted({r['goal_group'] for r in items if r['goal_group'] in groups}))
    first=rows[0]['messages'];instruction=first[0]['content'];ack=first[1]['content']
    assert all(x['messages'][0]['content']==instruction and x['messages'][1]['content']==ack for x in rows)
    out=ROOT/'docs/ee_sr_trial';out.mkdir(parents=True,exist_ok=True)
    atomic(out/'source_prompt.json',dict(instruction=instruction,acknowledgement=ack,source='published expert_sft.jsonl first two generic messages; no expert actions supplied to Actor',sha256=sequence_hash([instruction,ack])))
    atomic(out/'data_overlap.json',dict(expert_revision='7f1dfbe4f7a100ee1337843b1abbd2c25b8cf7ce',model_revision='4b8f0835a3f613e40541619462ae906db8dba95c',
        expert_file_sha256=file_hash(expert),agenttraj_source_sha256=file_hash(source),expert_dialogues=len(rows),source_dialogues=len(originals),
        identity_method='Exact full post-instruction user-observation sequence hashes; assistant content not used',
        unique_matches=unique_matches,goal_and_expert_action_sequence_reference_matches=policy_matches,policy_sequence_match_is_not_proof_of_hidden_variation_identity=True,ambiguous=ambiguous,unmatched_expert_rows=missing,partitions=partitions,
        checkpoint_actual_training_row_membership_verified=False,
        interpretation='Published SR recipe mixes expert+reflection; published expert corpus overlap is proven; exact checkpoint optimizer input manifest was not released/verified.',
        permitted_trial='retrospective Actor functionality only; no independent generalization or clean final-test performance claim',no_test_environment_queries=True,no_memory_or_threshold_updates=True))
    print(json.dumps(dict(expert=len(rows),unique_matches=len(unique_matches),ambiguous=len(ambiguous),unmatched=len(missing),
        partitions={k:dict(instances=v['instances'],exact_ids=len(v['unique_exact_dialogue_id_matches']),goal_overlap=len(v['normalized_goal_overlap_ids']),overlap_goals=len(v['overlapping_goal_groups'])) for k,v in partitions.items()}),indent=2))


if __name__=='__main__':main()
