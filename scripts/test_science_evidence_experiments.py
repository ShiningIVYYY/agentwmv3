import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.matrix import save,sha
import run_science_evidence_experiments as run
from prepare_science_evidence_dev import choose,TYPES


class SchedulingTests(unittest.TestCase):
    def matrix(self,root,protocol='science-evidence-v1'):
        manifest=root/'manifest.json';save(manifest,dict(tasks=[dict(task_id='dev')]))
        c=dict(env='scienceworld',manifest=str(manifest),url='http://router',model='ee-scienceworld-sr',
            tokenizer_path='/fixed/sr',tokenizer_url='http://tokenizer',science_url='http://science',
            max_steps=100,max_tokens=1024,generation_lock=None,science_context='task-anchored-v1',science_demo='/train/demo')
        dev=dict(c,manifest=str(root/'train.json'))
        job=dict(profile=run.PROFILE,backend='oracle',protocol=protocol,stage='stage1_guard',
            config=c,preflight_config=dev,workers=16,output=str(root/'oracle'),preflight=str(root/'smoke'))
        return dict(root=str(root),output=str(root),jobs={'stage1_guard/oracle':job},
            profiles={run.PROFILE:dict(env='scienceworld',task_ids=['dev'],wm_config='/wm.json')},
            development_task_ids=['train'],policy=dict(workers=6,smoke_limit=6,smoke_steps=6))

    def test_commands_only_change_policy_and_preserve_actor_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=self.matrix(Path(tmp))
            for protocol in ('science-evidence-v1','science-evidence-v2','science-evidence-v3','science-evidence-v3-obs'):
                m['jobs']['stage1_guard/oracle']['protocol']=protocol
                cmd=run.command(m,'stage1_guard/oracle')
                for key,value in [('--max-steps','100'),('--max-tokens','1024'),('--model','ee-scienceworld-sr'),
                    ('--workers','16'),('--science-context','task-anchored-v1'),('--reflection-max-rounds','3'),
                    ('--reflection-review-protocol',protocol)]:
                    self.assertEqual(cmd[cmd.index(key)+1],value)
                smoke=run.command(m,'stage1_guard/oracle',True)
                self.assertEqual(smoke[smoke.index('--manifest')+1],str(Path(tmp)/'train.json'))
                self.assertEqual(smoke[smoke.index('--max-steps')+1],'6')
                self.assertEqual(smoke[smoke.index('--workers')+1],'6')
                self.assertNotIn('--limit',cmd)

    def test_preregistered_stages_include_both_wms_and_observation_control(self):
        self.assertEqual(len(run.STAGES),4)
        self.assertEqual(sum(len(s[2]) for s in run.STAGES),7)
        for _,_,backends in run.STAGES[:3]:self.assertEqual(backends,('oracle','word2world'))
        self.assertEqual(run.STAGES[-1][1],'science-evidence-v3-obs')

    def test_maintenance_launch_preserves_original_attempt_count(self):
        matrix=dict(policy=dict(max_runner_launches=3))
        record=dict(launches=3)
        self.assertEqual(run.launch_limit(matrix,record,False),3)
        record['maintenance_restarts']=1
        self.assertEqual(run.launch_limit(matrix,record,False),4)
        self.assertEqual(record['launches'],3)
        self.assertEqual(run.launch_limit(matrix,record,True),1)

    def test_dev_selection_is_training_only_deterministic_and_excludes_demo(self):
        games=[dict(taskName=t,variationIdx=i) for t in TYPES for i in range(2)]
        training=[dict(item_id='sciworld_'+str(i)) for i in reversed(range(len(games)))]
        rows=choose(training,games,{'sciworld_0','sciworld_4509'})
        self.assertEqual(rows[0]['task_id'],'sciworld_1')
        self.assertEqual(len(rows),6)
        self.assertEqual([r['taskName'] for r in rows],list(TYPES))

    def fixture(self,root,protocol):
        m=self.matrix(root,protocol);job=m['jobs']['stage1_guard/oracle'];directory=Path(job['output'])
        actual=dict(job['config'],wm_backend='oracle',reflection_review_protocol=protocol,
            reflection_max_rounds=3,manifest_sha256=sha(root/'manifest.json'),execution=dict(workers=16),limit=None)
        save(directory/'config.json',actual)
        p=dict(action='move metal pot to red box',observation='You move the metal pot to the red box.',
            done=True,success=True,score=100)
        r=dict(review_protocol=protocol,rounds=0,initial_action=p['action'],final_action=p['action'],
            predictions=[p],reviews=[],prediction_exact_match=True,transition_metadata_visible=not protocol.endswith('-obs'),
            policy_events=[],memory_evidence=[])
        row=dict(task_id='dev',status='completed',done=True,score=100,success=True,task_description='REAL GOAL',
            steps=[dict(action=p['action'],reflection=r,messages=[dict(content='REAL GOAL')],prompt_tokens=100)])
        save(directory/'tasks/dev.json',row)
        sched=object.__new__(run.Scheduler);sched.matrix=m;sched.out=root
        return sched,directory,row

    def test_online_validator_rejects_successful_action_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched,d,row=self.fixture(Path(tmp),'science-evidence-v1')
            sched.validate('stage1_guard/oracle',False)
            s=row['steps'][0];s['action']='look around';s['reflection']['final_action']='look around'
            s['reflection']['predictions'].append(dict(action='look around',observation='Room'))
            save(d/'tasks/dev.json',row)
            with self.assertRaisesRegex(RuntimeError,'Successful action vetoed'):
                sched.validate('stage1_guard/oracle',False)

    def test_online_validator_rejects_privileged_metadata_in_obs_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched,d,row=self.fixture(Path(tmp),'science-evidence-v3-obs')
            packet=dict(original_forecast=dict(observation='Text',score=100))
            row['steps'][0]['reflection']['reviews']=[dict(messages=[dict(content='Rules\nINPUT:\n'+json.dumps(packet))],generation={})]
            save(d/'tasks/dev.json',row)
            with self.assertRaisesRegex(RuntimeError,'Privileged metadata leaked'):
                sched.validate('stage1_guard/oracle',False)


if __name__=='__main__':unittest.main()
