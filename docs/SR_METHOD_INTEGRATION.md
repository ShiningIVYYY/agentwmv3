# SR Actor/Reviewer integration and public-source data isolation

Status: implementation and19 CPU contracts complete; real constrained-model integration pending. This is not Phase6. User approved SR for BOTH Actor/Reviewer, shared weights, and paired9B/SR runs on a newly source-filtered cohort. The separate Phase5.9 External-only supplement uses the old cohort and original Reviewer; do not merge the two experiments.

## Frozen design

New opt-in profile `configs/evaluation/sr_isolated_integration_v1.json` inherits Phase5.10 bounded comparison settings: H20/nativecap21/history2/prompt4096,Actor response512,T=.4/p1/k0,BF16,seed0,ScienceWorld1.3.0,one-step independent-JVM Reset+Replay, one Reviewer round and compact-json-conservative-v2 prompt/cap192. Budgets Never0/Always20/Internal5/Full5. Same frozen3card/54case Memory, Gate rules, no Memory Correction or confirmed-action skip. The9B Actor/Reviewer configs and all old results remain unchanged. No Qwen2.5 learned WM is used.

The mandatory model-specific changes are the SR model/revision/tokenizer, format decoder and NEW NLL scoring identity/thresholds. Numeric9B thresholds cannot be reused with SR. Both models receive the same NEW calibration tasks and prespecified q=.75 distribution rule, and the same evaluation tasks. No task outcome selects thresholds. Each model's thresholds freeze before its evaluation. These are small development calibration/diagnostic samples, not stable population thresholds.

## Output adaptation

`FormatMachine` constrains exactly three Draft lines and the existing compact Reviewer JSON schema within a SINGLE generation. Labels/delimiters are forced syntax; all Premise,Consequence,Action values and Reviewer KEEP/REPLACE choices come from actual frozen-model token logits. There is no repair LLM, fake premise/feedback, second Actor call, expert demonstration or semantic rule choosing the action. Max512/192 and sampling unchanged. No retries; incomplete/invalid output is an explicit failure.

The raw model logits are retained BEFORE syntax masking and temperature, not renormalized grammar probabilities. Scored/unscored paths use the same constraint. The new model-side adapter is versioned and only enabled for SR;9B is unchanged. Therefore paired results compare complete configured policies, not a pure checkpoint-only intervention. A same-decoder9B factorial experiment has not been claimed.

Forced syntax can yield noncanonical BPE segmentation, e.g. a period then a separate newline instead of one merged token. A CPU test correctly rejected it under the historical exact-retokenization assertion. The new explicit `actual_generated_token_stream_offsets_v1` uses the ORIGINAL generated IDs and raw logprobs: verifies original prompt suffix, per-token/joint/full-prompt decoding equality, excludes special/empty/incomplete byte pieces, then maps those actual pieces to character intervals. It does not retokenize or rescore a different sequence. Historical alignment remains intact. All canonical-retokenization differences are reported rather than hidden.

Local=max(mean actual-token Premise-value NLL,mean Expected-Consequence-value NLL). Global=mean all response-token raw NLL including forced labels/Action, excluding prompt/EOS/padding. Fixed syntax can influence Global; scores remain uncalibrated generation statistics, not action-error probabilities. A real raw-forward-hook check, repeated NLL and scored/unscored token parity are required before deployment. Backend `hf_ee_sr_structured_raw_bf16_v1` has independent model/config identity. No old thresholds apply.

## Data isolation

Pinned corpus revision `7f1dfbe4f7a100ee1337843b1abbd2c25b8cf7ce`: full expert2038rows/18,808,399bytes/SHA256096895094fe44005b03037b44414388bc4935dcc0224fcc678f9811545cc2708 and reflection35065rows/271,777,648bytes/SHA2568020d0c6fe9dd86105870fc4bfe7c9ec45fcedb555318c9979c0e485016eb4ba. Both contain547 normalized goals. Only these public SR-recipe sources are used; no IWM model/data or training is needed.

Audit repair: the previous helper recognized only `This room is called`, retaining outside observations inside some goal hashes. The new parser covers BOTH room and outside-location boundaries and rejects unknown formats. First strict audit stopped on this case before model experiments; no incomplete audit is accepted as isolation. Old trial reports remain historical and must not substitute for this full source audit.

Exclude all200 final-test IDs/goals, all original197 development instances/goals, all known prior task exposure, and ALL frozen Memory source IDs/goals. Reserve whole previously unused TRAIN goal groups for a separate development extension; never ingest them into Memory later.64 candidates/18 goals remain after public-source filtering. Deterministic family-round-robin/SHA256 ranking selects6 calibration goals and3 evaluation goals, one instance per goal; selection was frozen before outcome generation. The original split and frozen Memory are not overwritten; the separate reservation manifest governs this extension.

Calibration IDs:4143,1637,4140,1630,4148,1658. Evaluation IDs:4151,1633,4150. Every row retains native task name,Variation,original partition and reserved role in `docs/sr_integration/isolation_manifest.json`. No semantic-template independence is claimed; the cohort is narrow chemistry/conductivity tasks. Exact normalized goals are disjoint from BOTH complete public sources, but the checkpoint's unpublished optimizer-input manifest is unavailable. Full checkpoint-training isolation therefore remains UNVERIFIED. No claim of clean unseen-template/general benchmark accuracy is justified.

## Commands and limits

```bash
.venv-qwen35-clean/bin/python scripts/download_sr_reflection_audit.py
.venv-qwen35-clean/bin/python scripts/prepare_sr_isolation.py
.venv-qwen35-clean/bin/python scripts/test_sr_integration.py
CUDA_VISIBLE_DEVICES=1 .venv-qwen35-clean/bin/python scripts/run_sr_method.py --model sr --run outputs/sr_method_integration_v1
CUDA_VISIBLE_DEVICES=1 .venv-qwen35-clean/bin/python scripts/run_sr_method.py --model qwen35 --run outputs/qwen35_isolated_control_v1
```

Per model:6 calibration tasks×up to8actions,3 evaluation tasks×seed0×4policies=12episodes, plus2 charged validation generations; caps1Mtokens/700model calls/6000transitions/40000capture bundles/1h. No full200×8 rollout. Owned JVMs close in finally. Baseline methods/sources are hash guarded. Original missing/failed attempts and costs are retained, never replaced with success-shaped records. Real results, test commands, costs and final commit will be appended after completion.
