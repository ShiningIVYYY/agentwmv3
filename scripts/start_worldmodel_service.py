#!/usr/bin/env python3
"""Start one audited Word2World server on an explicitly available GPU."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from urllib.parse import urlparse
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldmodel.backends import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', choices=['alfworld', 'scienceworld', 'webshop'], required=True)
    parser.add_argument('--config')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--vllm', default='/home/shn/miniconda3/envs/verl-agent/bin/vllm')
    args = parser.parse_args()
    config = load_config(args.config or ROOT / 'configs/worldmodel' / (args.env + '.json'), ROOT, args.env)
    url = urlparse(config['url'])
    if url.hostname != '127.0.0.1':
        raise RuntimeError('Local service launcher requires a 127.0.0.1 endpoint')
    try:
        result = requests.get(config['url'] + '/v1/models', timeout=3)
        if result.ok:
            assert any(m['id'] == config['served_model_name'] and
                Path(m.get('root', '')).resolve() == Path(config['checkpoint']).resolve() and
                m.get('max_model_len') == config['max_model_len']
                for m in result.json()['data']), 'Port occupied by a different checkpoint/context config'
            print('Already ready:', config['url'])
            return
    except requests.RequestException:
        pass
    with socket.socket() as sock:
        if sock.connect_ex((url.hostname, url.port)) == 0:
            raise RuntimeError('Port occupied but model service is not ready')
    gpu = args.gpu if args.gpu is not None else config['gpu']
    mem = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
    used, total = [int(s.strip()) for s in mem.split(',')]
    required = int(total * config['gpu_memory_utilization'])
    if total - used < required:
        raise RuntimeError('GPU %d has %d/%d MiB free; deployment needs %d MiB. '
                           'Select a free GPU; this script does not stop other services.' %
                           (gpu, total-used, total, required))
    logs = ROOT / 'outputs/wm_reflection_setup'
    logs.mkdir(parents=True, exist_ok=True)
    command = [args.vllm, 'serve', config['checkpoint'], '--host', '127.0.0.1', '--port', str(url.port),
        '--served-model-name', config['served_model_name'], '--tensor-parallel-size', '1',
        '--pipeline-parallel-size', '1', '--gpu-memory-utilization', str(config['gpu_memory_utilization']),
        '--max-model-len', str(config['max_model_len']), '--max-num-seqs', '8', '--enable-prefix-caching']
    process = subprocess.Popen(command, cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1'),
        stdin=subprocess.DEVNULL, stdout=open(logs / (args.env + '_server.log'), 'a'),
        stderr=subprocess.STDOUT, start_new_session=True)
    (logs / (args.env + '_server.json')).write_text(json.dumps(dict(pid=process.pid, gpu=gpu,
        command=command, config=config), indent=2))
    print('Started', process.pid, 'on GPU', gpu, '; log:', logs / (args.env + '_server.log'))


if __name__ == '__main__':
    main()
