#!/usr/bin/env python3
"""Paper-style single-agent evaluator for ALFWorld, ScienceWorld, and WebShop.

The model is served separately by vLLM's OpenAI-compatible server.  This keeps
the agent environments CPU-side and makes every prompt/action/transition auditable.
"""
import argparse, json, os, re, sys, time
from pathlib import Path
import requests

ALF = """You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task}\nPrior to this step, you have already taken {step_count} step(s). Below are the most recent 2 observations and the corresponding actions you took: {history}\nYou are now at step {step} and your current observation is: {obs}\nYour admissible actions of the current situation are: [{actions}].\n\nNow it's your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""
SCI = """You are an expert agent operating in the ScienceWorld environment. Your task is: {task}\nYour current observation is: {obs}\nYour admissible actions are: [{actions}].\nReason step-by-step in <think> tags, then output exactly one admissible action in <action> tags."""
WEB = """You are an expert autonomous agent operating in the WebShop e-commerce environment.\nYour task is to: {task}.\nPrior to this step, you have already taken {step_count} step(s). Below are the most recent 2 observations and the corresponding actions you took: {history}\nYou are now at step {step} and your current observation is: {obs}.\nYour admissible actions of the current situation are: [{actions}].\n\nNow it's your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

def complete(url, model, prompt, temperature, disable_thinking=False):
    payload={
        'model': model, 'messages': [{'role': 'user', 'content': prompt}],
        'temperature': temperature, 'max_tokens': 1024, 'seed': 0,
    }
    if disable_thinking:
        payload['chat_template_kwargs']={'enable_thinking': False}
    r = requests.post(url.rstrip('/') + '/v1/chat/completions', json=payload, timeout=600)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']

def action(text, admissible):
    m = re.findall(r'<action>\s*(.*?)\s*</action>', text, flags=re.S | re.I)
    lines=text.strip().splitlines()
    raw = m[-1].strip() if m else (lines[-1].strip() if lines else '')
    # Some base checkpoints use the equally common Markdown-like form
    # ``[action] ...``.  Treat both formats identically rather than feeding an
    # invalid string to the environment.
    raw = re.sub(r'^\s*\[/?action\]\s*', '', raw, flags=re.I)
    raw = re.sub(r'\s*\[/?action\]\s*$', '', raw, flags=re.I).strip()
    low = {str(x).strip().lower(): x for x in admissible}
    if raw.lower() in low:
        return low[raw.lower()], True
    # WebShop's search field is intentionally open-ended; the placeholder in
    # ``admissible`` describes its grammar, not a literal action.
    if re.fullmatch(r'search\[[^\]]+\]', raw, flags=re.S | re.I):
        return raw, True
    return raw, False

def write(out, obj):
    with open(out / 'trajectories.jsonl', 'a') as f: f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def task_ids(args):
    if not args.task_ids_file:
        return list(range(args.tasks))
    rows=json.load(open(args.task_ids_file))
    ids=[]
    for row in rows:
        value=row.get('item_id', row) if isinstance(row, dict) else row
        m=re.search(r'(\d+)$', str(value))
        if not m: raise ValueError(f'Cannot parse task id: {value!r}')
        ids.append(int(m.group(1)))
    return ids

def alf(args, out):
    import yaml
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    cfg_path = os.path.expanduser('~/.cache/alfworld/alfworld/configs/base_config.yaml')
    os.environ['ALFWORLD_DATA'] = os.path.expanduser('~/.cache/alfworld')
    cfg = yaml.safe_load(open(cfg_path)); cfg['dataset']['num_eval_games'] = args.tasks
    base = AlfredTWEnv(cfg, train_eval='eval_in_distribution'); env = base.init_env(1)
    for i in range(base.num_games):
        obs, info = env.reset(); steps=[]; won=False
        task=re.search(r'Your task is to:\s*(.*)', obs[0]).group(1) if 'Your task is to:' in obs[0] else ''
        for t in range(args.max_steps):
            acts = info['admissible_commands'][0]
            hist='; '.join(f'Observation: {s["observation"]} Action: {s["action"]}' for s in steps[-2:]) or 'None'
            p = ALF.format(task=task, step_count=len(steps), history=hist, step=t+1, obs=obs[0], actions=', '.join(acts))
            text=complete(args.url,args.model,p,args.temperature,args.disable_thinking); a,valid=action(text,acts)
            prior_obs=obs[0]; obs, score, done, info=env.step([a]); steps.append({'prompt':p,'output':text,'action':a,'valid':valid,'observation':prior_obs})
            won=bool(info.get('won',[False])[0]);
            if done[0]: break
        write(out, {'task_id':i,'success':won,'steps':steps})
    env.close()

def science(args, out):
    base=args.science_url.rstrip('/')
    for i in task_ids(args):
        eid=requests.post(base+'/create',timeout=180).json()['id']
        rst=requests.post(base+'/reset',json={'id':eid,'data_idx':i},timeout=180).json(); steps=[]; done=False
        for t in range(args.max_steps):
            acts=requests.get(base+'/admissible_actions',params={'id':eid},timeout=180).json()['admissible_actions']
            # The raw ScienceWorld API also exposes hundreds of graph-only
            # ``connect``/``disconnect`` combinations.  They consume the
            # prompt context and mask normal task actions (open/go/pick up),
            # although they are never part of the standard text interface.
            shown_acts=[a for a in acts if not a.startswith(('connect ', 'disconnect '))]
            if not shown_acts:
                shown_acts=acts
            p=SCI.format(task=rst.get('task_description',''),obs=rst.get('observation',''),actions=', '.join(shown_acts))
            text=complete(args.url,args.model,p,args.temperature,args.disable_thinking); a,valid=action(text,acts)
            rst=requests.post(base+'/step',json={'id':eid,'action':a},timeout=180).json(); steps.append({'prompt':p,'output':text,'action':a,'valid':valid})
            done=bool(rst.get('done')); 
            if done: break
        write(out, {'task_id':i,'success':float(rst.get('score',0))>=1,'score':rst.get('score',0),'steps':steps})
        requests.post(base+'/close',json={'id':eid},timeout=180)

def webshop(args, out):
    # Keep this runner self-contained when invoked from the repository root:
    # WebShop is vendored by Verl-Agent rather than installed as a wheel.
    webshop_root = Path(__file__).resolve().parents[1] / 'external/verl-agent/agent_system/environments/env_package/webshop/webshop'
    if str(webshop_root) not in sys.path:
        sys.path.insert(0, str(webshop_root))
    # pyjnius discovers Java through JAVA_HOME when the conda environment is
    # invoked by its absolute Python path (rather than through ``conda run``).
    os.environ.setdefault('JAVA_HOME', str(Path(sys.executable).resolve().parents[1]))
    os.environ['PATH'] = os.environ['JAVA_HOME'] + '/bin:' + os.environ.get('PATH', '')
    os.environ.setdefault('JVM_PATH', os.environ['JAVA_HOME'] + '/lib/jvm/lib/server/libjvm.so')
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    env=WebAgentTextEnv(observation_mode='text', num_products=None, seed=0)
    for i in task_ids(args):
        obs=env.reset(session=i); steps=[]; done=False
        # The simulator embeds the assigned instruction in the text state.
        # ``instruction_text`` is not a public server attribute and is empty.
        task=''
        m=re.search(r'Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]', str(obs), flags=re.S)
        if m: task=m.group(1)
        for t in range(args.max_steps):
            acts=env.get_available_actions()['clickables']; admissible=['search[<keywords>]']+['click['+x+']' for x in acts]
            hist='; '.join(f'Observation: {s["observation"]} Action: {s["action"]}' for s in steps[-2:]) or 'None'
            p=WEB.format(task=task,step_count=len(steps),history=hist,step=t+1,obs=env.observation,actions=', '.join(admissible))
            text=complete(args.url,args.model,p,args.temperature,args.disable_thinking); a,valid=action(text,admissible)
            state,reward,done,_=env.step(a); steps.append({'prompt':p,'output':text,'action':a,'valid':valid,'reward':reward,'observation':str(env.observation)})
            if done: break
        write(out, {'task_id':i,'success':bool(done and reward>0),'reward':reward if steps else 0,'steps':steps})

def main():
 p=argparse.ArgumentParser(); p.add_argument('--env',choices=['alfworld','scienceworld','webshop'],required=True); p.add_argument('--url',required=True); p.add_argument('--model',required=True); p.add_argument('--output',required=True); p.add_argument('--tasks',type=int,default=2); p.add_argument('--task-ids-file',help='JSON list of official benchmark item_id values (e.g. sciworld_3491).'); p.add_argument('--max-steps',type=int,default=50); p.add_argument('--temperature',type=float,default=.4); p.add_argument('--science-url',default='http://127.0.0.1:36001'); p.add_argument('--disable-thinking',action='store_true',help='Disable server-side hidden reasoning (needed by Qwen3.5 Transformers server).'); args=p.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True); json.dump(vars(args),open(out/'config.json','w'),indent=2)
 {'alfworld':alf,'scienceworld':science,'webshop':webshop}[args.env](args,out)
if __name__=='__main__': main()
