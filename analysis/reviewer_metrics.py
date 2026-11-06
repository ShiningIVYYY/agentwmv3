"""Posthoc native outcome labels and strict within-JVM branch prerequisites."""


def termination_causes(native):
    causes=[]
    for flag,label in [('negative_score_termination','negative_native_score'),
                       ('simulator_completed','simulator_completed'),
                       ('sdk_clock_limit_exceeded','SDK_simulator_clock_limit'),
                       ('horizon_truncated','actor20_turn_cap')]:
        if native[flag]:causes.append(label)
    return causes


def feedback_kind(observation,score_delta):
    if score_delta<0:return 'observed_task_score_decrease'
    if 'No known action matches' in observation:return 'parser_or_object_resolution_failure'
    if observation.strip()=='The door is not open.':return 'state_precondition_failure'
    if observation.startswith(('You focus on','You open','This room is called')):return 'accepted_feedback_not_proof_of_task_success'
    return 'unknown'


def verified_pair_utility(skip,call):
    for item in (skip,call):
        if item.get('restore_provenance',{}).get('heap_matches') is not True:
            raise ValueError('Observable similarity alone is not a verified counterfactual')
    a,b=skip['restore_provenance'],call['restore_provenance']
    if a['snapshot_id']!=b['snapshot_id']:raise ValueError('Different snapshots')
    for key in ('task_id','source_step','remaining_actor_budget','initial_score','initial_simulator_moves'):
        if skip[key]!=call[key]:raise ValueError('Unmatched branch '+key)
    return call['final_score']-skip['final_score']
