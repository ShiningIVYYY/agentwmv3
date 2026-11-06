"""Independent diagnostic variants; the original reviewer remains untouched."""
import copy


def official_validity(action, actions):
    # Presence supplies API enumeration evidence, not guaranteed task utility.
    # The SDK list is incomplete: absence must remain Unknown.
    return 'api_enumerated' if action in actions else 'unknown'


class ControllerWrapper:
    def __init__(self, original):self.original=original
    def __getattr__(self, name):return getattr(self.original,name)
    def commit(self, action, actual, log):return self.original.commit(action,actual,log)


class PredictionOnly(ControllerWrapper):
    def select(self,candidate,observation,steps,actions,step_number):
        prediction=self.wm.predict(candidate)
        self.recorder('diagnostic_prediction_only',step_number,dict(action=candidate,prediction=prediction))
        return candidate,dict(backend=self.wm.backend,review_protocol='json',initial_action=candidate,
            final_action=candidate,reviews=[],predictions=[dict(action=candidate,**prediction)],rounds=0,
            replacements=0,invalid_reviews=0,hit_round_limit=False,diagnostic_variant='prediction_only',
            action_modification_enabled=False)


class ValidityChecked(ControllerWrapper):
    """Run the ORIGINAL reviewer; separately guard its final proposed replacement.

    No new prompt, retry, generated action or feedback lookup. Unknown validity
    retains the original draft; it is never relabelled as proven invalid/valid.
    """
    def select(self,candidate,observation,steps,actions,step_number):
        proposed,log=self.original.select(candidate,observation,steps,actions,step_number)
        status=official_validity(proposed,actions)
        blocked=proposed!=candidate and status=='unknown'
        log.update(diagnostic_variant='validity_checked',proposed_final_action=proposed,
            official_action_validity=status,guard_blocked_unknown_replacement=blocked,
            guard_information='current pre-action SDK enumeration only; absence is Unknown',
            guard_original_draft_validity=official_validity(candidate,actions),
            semantic_utility_not_certified=True)
        if blocked:
            proposed=candidate
            # The draft was already predicted from this exact unchanged prefix.
            # Reuse its logged prediction for commit comparison without a new call.
            if log['predictions'][-1]['action']!=candidate:
                cached=copy.deepcopy(log['predictions'][0]);cached.update(cached=True,
                    diagnostic_cache_reason='guard reverted to already forecast draft at same real prefix')
                log['predictions'].append(cached)
            log['final_action']=candidate
        return proposed,log
