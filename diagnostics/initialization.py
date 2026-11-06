"""Validated WM-only initialization. Does not patch the upstream collector.

Root ObjectTree children include electrical terminals, not only locations.
Each claimed room must be observed after teleport; every probe starts from
the same helper-world heap. Actual Agent/Reviewer never see this collector.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

VERSION='validated-room-initializer-v1'


def observed_room(text):
    found=re.match(r'^This (?:room|outside location) is called the ([^.]+)\.',text)
    return found.group(1) if found else None


def validated_room_view(name,text):return observed_room(text)==name


def teleport_command(name,official_actions):
    command='teleport to '+name
    return command if command in official_actions else None


def collect(config,row,root):
    from scienceworld import ScienceWorldEnv
    from worldmodel.contexts import collector
    from worldmodel.science_snapshot import load_bridge
    from evaluation.profile import canonicalize_observation
    source=collector(config,'sciworld');env=ScienceWorldEnv();loader=None
    try:
        env.load(row['taskName'],row['variationIdx'],'teleportAction')
        snapshot=source.get_full_snapshot(env)
        candidates=sorted({v['name'] for v in (snapshot['object_tree'].get('contents') or {}).values() if isinstance(v,dict) and isinstance(v.get('name'),str)})
        bridge,loader=load_bridge(env,Path(root));python={k:copy.deepcopy(v) for k,v in env.__dict__.items() if k not in ('_gateway','server','_obj_tree_tempdir')}
        heap=bridge.take(env.server);views=[];excluded=[]
        for name in candidates:
            command=teleport_command(name,snapshot.get('valid_action_object_combinations') or [])
            if command is None:
                excluded.append(dict(candidate=name,reason='no_official_teleport_command'))
                continue
            heap.restore();env.__dict__.update(copy.deepcopy(python))
            feedback,reward,done,info=env.step(command)
            text=env.look();actual=observed_room(text)
            if not validated_room_view(name,text):
                raise ValueError('Official teleport failed room-label validation: '+name+' -> '+str(actual))
            views.append(dict(room=name,look=canonicalize_observation(text),possible_objects=sorted(env.get_possible_objects()),
                simulator_moves_after_teleport=env.get_num_moves(),verified_room_label=True))
        if not views:raise ValueError('No verified room views')
        parts=['\n=== Goal Progress ===\n'+str(snapshot.get('goal_progress') or '')+'\n','\n=== Possible Actions ===\n']
        parts += [' - '+action+'\n' for action in snapshot.get('possible_actions') or []]
        parts += ['\n=== Per-Room Observations ===\n']
        for view in views:parts.append('== Room: '+view['room']+' ==\n'+view['look'].strip()+'\nPossible Objects: '+', '.join(view['possible_objects'])+'\n\n')
        description=''.join(parts)
        return dict(version=VERSION,task_id=row['task_id'],taskName=row['taskName'],variationIdx=row['variationIdx'],
            description=description,description_sha256=hashlib.sha256(description.encode()).hexdigest(),
            candidates=candidates,verified_rooms=[v['room'] for v in views],excluded=excluded,views=views,
            scope='WM-only helper world; each room view follows one real teleport from restored helper baseline; not an Actor future rollout')
    finally:
        if loader is not None:loader.close()
        env.close()


class ValidatedWorldModelSetup:
    """Same inference interface; consume a source-validated frozen description."""
    def __init__(self,science_python,root,output):
        from gates.worldmodel_setup import ExistingWorldModelSetup
        self.base=ExistingWorldModelSetup(science_python,root,output)
        self.root,self.output=Path(root),Path(output);self.costs=[]
    def count(self,config,messages):return self.base.count(config,messages)
    def context(self,config,row,task,initial):
        started=time.perf_counter();path=self.root/'outputs/phase55_setup/validated_initializers'/(row['task_id']+'.json')
        value=json.loads(path.read_text())
        if value['version']!=VERSION or (value['taskName'],value['variationIdx'])!=(row['taskName'],row['variationIdx']):raise ValueError('Initializer identity mismatch')
        description=value['description']
        if hashlib.sha256(description.encode()).hexdigest()!=value['description_sha256']:raise ValueError('Initializer description changed')
        context='# Environment Information (Only visible to Assistant)\n\n'+description+'\n\n# User Environment Information (Displayed to User)\n\n'+task+'\n'+initial['wm_observation']+'\n'
        dest=self.output/('wm_context_'+row['task_id']);dest.mkdir()
        (dest/'context.json').write_text(json.dumps(context))
        self.costs.append(dict(task_id=row['task_id'],seconds=time.perf_counter()-started,characters=len(context),only_visible_to_wm=True,
            method=VERSION,description_sha256=value['description_sha256'],source=str(path),offline_collection_separate=True))
        return context
