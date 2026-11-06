#!/usr/bin/env python3
"""CPU-only causal reproducer of ambiguous non-room poisoning a later teleport."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scienceworld import ScienceWorldEnv
from worldmodel.science_snapshot import load_bridge


def main():
    e=ScienceWorldEnv();loader=None
    try:
        e.load('measure-melting-point-known-substance',82,'teleportAction');e.step('look around')
        bridge,loader=load_bridge(e,ROOT);state={k:copy.deepcopy(v) for k,v in e.__dict__.items() if k not in ('_gateway','server','_obj_tree_tempdir')};heap=bridge.take(e.server)
        ambiguous=e.step('teleport terminal 1')[0];following=e.step('teleport workshop')[0];poisoned=e.look()
        heap.restore();e.__dict__.update(copy.deepcopy(state));clean=e.step('teleport to workshop')[0];recovered=e.look()
        result=dict(task_id='sciworld_742',offline_helper_only=True,ambiguous_first_response=ambiguous,
            legacy_following_teleport_response=following,legacy_following_room_look=poisoned,
            restored_teleport_response=clean,restored_room_look=recovered,
            failed_command_poisoning_reproduced='workshop' not in poisoned.splitlines()[0] and 'workshop' in recovered.splitlines()[0])
        if not result['failed_command_poisoning_reproduced']:raise ValueError('Failure mechanism not reproduced')
        out=ROOT/'outputs/phase55_setup/initializer_failure_probe.json';out.write_text(json.dumps(result,indent=2)+'\n');print('reproduced',True)
    finally:
        if loader is not None:loader.close()
        e.close()


if __name__=='__main__':main()
