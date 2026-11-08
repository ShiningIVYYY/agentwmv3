#!/usr/bin/env python3
"""Version, strict state comparison and configuration guards; no fake metrics."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from worldmodel.ground_truth_environment import simulator_identity,state_fingerprint,SIMULATOR_PROFILES
from reflection.replay_review import review_messages
from scripts.run_phase57_replay import load_calibration_source

class Profiles(unittest.TestCase):
    def test_legacy_default_kept(self):
        _,identity=simulator_identity({})
        self.assertEqual(identity['version'],'1.2.3')

    def test_official_manifest_and_pinned_binary(self):
        path,identity=simulator_identity({'simulator_profile':'scienceworld130'})
        with zipfile.ZipFile(path) as z:
            manifest=z.read('META-INF/MANIFEST.MF').decode()
        self.assertIn('Specification-Version: 1.3.0',manifest)
        self.assertEqual(identity['upstream_commit'],'e8216d6044e8e39be9fcb185e3b2dfb602584b52')

    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):simulator_identity({'simulator_profile':'guess'})

    def test_tampered_jar_rejected(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b'not official physics');f.flush()
            with patch.dict(os.environ,{'SCIENCEWORLD_REPLAY_JAR':f.name}):
                with self.assertRaises(ValueError):simulator_identity({'simulator_profile':'scienceworld130'})

    def test_fingerprint_preserves_physics_rng_and_reward(self):
        s=dict(transition=dict(raw_observation='x',observation='x',reward=0,score=0,done=False,moves=2),
               object_tree_sha256='a',selected_rng={'seed':1},simulator={'version':'1.3.0'})
        for key,value in [('object_tree_sha256','b'),('selected_rng',{'seed':2}),('simulator',{'version':'1.2.3'})]:
            other=copy.deepcopy(s);other[key]=value
            self.assertNotEqual(state_fingerprint(s),state_fingerprint(other))
        other=copy.deepcopy(s);other['transition']['reward']=1
        self.assertNotEqual(state_fingerprint(s),state_fingerprint(other))
        other=copy.deepcopy(s);other['transition']['raw_observation']='different presentation'
        self.assertEqual(state_fingerprint(s),state_fingerprint(other))

    def test_private_next_state_hash_not_in_reviewer(self):
        result=dict(status='ok',prediction_available=True,predicted_observation='door opens',
            predicted_reward=0,predicted_done=False,predicted_score=0,candidate_state_sha256='SECRET_AUDIT',
            simulator={'upstream_commit':'SECRET_AUDIT'})
        self.assertNotIn('SECRET_AUDIT',json.dumps(review_messages('goal','obs',[],'wait',result)))

    def test_old_threshold_collection_rejected_for_new_engine(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name);(p/'manifest.json').write_text(json.dumps(dict(pilot_profile={})))
            with self.assertRaisesRegex(ValueError,'simulator profile mismatch'):
                load_calibration_source(p,{},[],simulator_profile='scienceworld130')

if __name__=='__main__':unittest.main()
