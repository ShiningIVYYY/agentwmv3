"""Materialize the EXISTING single-run episode body for scheduling-only reuse."""
import ast,hashlib,json,sys,textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def extracted(tree):
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    definitions={n.name:n for n in ast.walk(main) if isinstance(n,ast.FunctionDef)}
    helpers=[definitions[n] for n in ['generate','new_env','formal_step']]
    policy=next(n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='policy')
    return helpers,policy.body


def validate(source,materialized):
    helpers,body=extracted(ast.parse(source))
    function=next(n for n in ast.parse(materialized).body if isinstance(n,ast.FunctionDef) and n.name=='run_episode')
    expected=helpers+body
    if ast.dump(ast.Module(body=function.body[:-1],type_ignores=[]),include_attributes=False)!=ast.dump(ast.Module(body=expected,type_ignores=[]),include_attributes=False):
        raise ValueError('Parallel episode differs from original runtime AST')
    if not isinstance(function.body[-1],ast.Return):raise ValueError('Missing result return')
    return hashlib.sha256(ast.dump(ast.Module(body=expected,type_ignores=[]),include_attributes=False).encode()).hexdigest()


def main():
    source=(ROOT/'scripts/run_alfworld_pilot.py').read_text();lines=source.splitlines(keepends=True);helpers,body=extracted(ast.parse(source))
    chunks=[]
    for n in helpers:chunks.append(textwrap.dedent(''.join(lines[n.lineno-1:n.end_lineno])))
    chunks.append(textwrap.dedent(''.join(lines[body[0].lineno-1:body[-1].end_lineno])))
    content='''"""Generated verbatim episode logic; scheduling only. Do not edit by hand."""
import time
from scripts.run_alfworld_pilot import (atomic,append,AlfEnvironment,AlfGroundTruthWM,RuleGate,GateInput,ExternalInput,
    draft_messages,review_messages,parse_review,request_seed,fields,file_hash,fingerprint)


def run_episode(*,row,policy,seed,actor,memory,gate_config,p,out,a,ledger,native_goals,episodes):
'''+''.join(textwrap.indent(chunk,'    ')+'\n' for chunk in chunks)+'    return result\n'
    digest=validate(source,content)
    (ROOT/'evaluation/alfworld_episode.py').write_text(content)
    (ROOT/'docs/alfworld_pilot/parallel_episode_identity.json').write_text(json.dumps(dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),episode_ast_sha256=digest,body_and_helpers_identical=True),indent=2)+'\n')
    print(digest)


if __name__=='__main__':main()
