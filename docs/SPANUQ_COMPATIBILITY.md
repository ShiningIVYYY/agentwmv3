# SpanUQ Compatibility Investigation — Phase 1B

**Conclusion B: SpanUQ remains unavailable; recommend the task-specific training-free fixed-span token-NLL fallback as the next route, subject to user selection.** Phase 1B investigation/planning is complete; SpanUQ Phase 1 acceptance is still not passed. No Phase 2 or benchmark work is authorized by this document.

## Scope and latest official check

Source: [official GitHub](https://github.com/damon-demon/SpanUQ); [public Issues](https://github.com/damon-demon/SpanUQ/issues?q=is%3Aissue); [PRs](https://github.com/damon-demon/SpanUQ/pulls?q=is%3Apr); [releases](https://github.com/damon-demon/SpanUQ/releases); [HF model/discussions](https://huggingface.co/DamonDemon/SpanUQ/discussions).

The new check fetched official refs and source again, retrieved HF head/history/tree/discussions, and inspected public GitHub issue/PR/release HTML. See phase1b/official_recheck.json, source_summary.json, github_refs.txt and public_html_summary.json for timestamps, URLs and access errors. No Issue/PR/message was posted.

Observed:
- GitHub main remains 778b1c1e3eeeda66233fe6b426f9303f4c7f76a3. Advertised refs contain only HEAD and main; no public repair branch/tag/pull ref was advertised. The four-commit code history and identical model/config/producer blobs documented in Phase 1 remain applicable.
- GitHub Issues and PR HTML show zero results; releases HTML says there are no releases. REST issue/branch/tag endpoints returned 403 rate-limit errors, so they were not counted as successful API searches. Public HTML and git refs are the fallback evidence, not proof about private/unpublished/deleted work.
- HF main remains 97ca9c820ac3561259956ec21cbdd984c3132d37. Qwen3-4B probe still has LFS SHA256 78a203415037cb6645385d27b14d4207d184d522cc2c33af07098f2f2254fb12. Its last weight/config upload is 68053f75e53843b70015473c981e5cca7d14869f.
- HF lists one PR, number 1, titled Update pipeline tag and add project page link. It is metadata-only, not a matching model implementation. No new compatible release was found in the resources inspected.

## Complete state/schema comparison

The read-only script scripts/investigate_spanuq_structure.py inventories every key, shape and dtype in the real checkpoint and official instantiated model. phase1b/structural_comparison.json contains all 134 union entries, finite checks and before/after SHA256.

| Difference | Actual evidence | Classification / recovery judgment |
| --- | --- | --- |
| u_conditioner.cond_mlp.0.weight | [256,3] | Undeclared model component; public source has no matching module or forward |
| u_conditioner.cond_mlp.0.bias | [256] | Same component |
| u_conditioner.cond_mlp.2.weight | [512,256] | Same component |
| u_conditioner.cond_mlp.2.bias | [512] | Same component |
| token_encoder.pos_enc.pe | checkpoint [1,16384,512], model [1,4096,512] | Deterministic persistent buffer extent differs; constructor/config exposure is missing |
| mob_k=3 | source field mob_n_components; two real MoB output matrices have 9 output rows | Explicit legacy naming plus an architecture-affecting K setting; dropping it incorrectly selects default K=1 |
| seq_agg_mode=importance | source config calls it seq_aggregation | Naming discrepancy, but the config value is not forwarded by model_kwargs; actual public forward uses importance aggregation directly |
| max_seq_len=2048 | public config rejects it; real PE has length 16384 | Input-length metadata and positional-buffer extent are separate; 2048 must not be assumed to construct the PE buffer |

Counts: checkpoint 134 tensors, public model 130; no public-model key is missing from checkpoint; 129 common tensors have matching shapes; one common buffer has a shape mismatch; four additional conditioner keys remain. All checkpoint tensors are finite. Conditioner norms are nonzero, but that does not establish how or whether they are used at inference. Parameter shapes alone cannot identify the three input features, nonlinearities, connection point, gating or training-only status.

### New PE value verification

Using the unmodified official SinusoidalPositionalEncoding class independently with max_len=16384, its entire generated buffer equals the checkpoint buffer elementwise: maximum absolute error 0. The checkpoint prefix of length 4096 also equals the current model buffer exactly, maximum error 0.

This is stronger evidence that PE is a deterministic length/configuration difference rather than an unknown learned positional algorithm. It does NOT make the checkpoint loadable: no original tensor was cropped/replaced/modified, no buffer in the model was patched, and the four absent conditioner tensors still have no verified forward contract.

### Explicit aliases do not restore the architecture

Phase 1 records mob_k -> mob_n_components and seq_agg_mode -> seq_aggregation, plus max_seq_len as a separately recorded response validation cap. Other unknown fields are rejected. These are diagnostic translations, not a claim to recover the missing implementation. Actual strict=True recheck still exits 2 and saves all four unexpected keys and the PE mismatch in phase1b/strict_recheck.json.

No strict=False, key deletion, unused dummy module, guessed conditioner forward, checkpoint rewriting or parameter training was used. Known deterministic PE values cannot justify ignoring the unknown component. There is presently no reliable full recovery route from the inspected public artifacts.

## What official evidence would unblock recovery?

Request the exact implementation/commit used to export this weight SHA, including u_conditioner inputs and forward path, PE maximum-length constructor/config, mapping of legacy field names, original Qwen3-4B/tokenizer revisions and zero-based decoder-block convention. If the module is training-only, an author-supplied executable inference/export path is needed, not our inference from tensor names.

A new official implementation must be separately pinned and validated with strict=True, all key/shape checks, original assets unchanged, actual hidden alignment and real span/score/sequence/calibration tests. No matching official code was found, so new SpanUQ inference was not attempted after the expected strict failure.

## Reproduction

Run from agentwmv3 with the existing isolated environment and pinned assets:

~~~bash
git -C external/SpanUQ ls-remote origin
git -C external/SpanUQ fetch origin
CUDA_VISIBLE_DEVICES= .venv-spanuq/bin/python scripts/investigate_spanuq_structure.py --source external/SpanUQ --checkpoint-dir checkpoints/spanuq/Qwen3-4B --output outputs/phase1b/structural_comparison.json
CUDA_VISIBLE_DEVICES= .venv-spanuq/bin/python scripts/check_spanuq_compatibility.py --source external/SpanUQ --checkpoint-dir checkpoints/spanuq/Qwen3-4B --output outputs/phase1b/strict_recheck.json
~~~

The structure command exits 0 (successful investigation); strict loading exits 2 (model incompatibility). This is not successful uncertainty inference. No new dependencies or weights were installed/downloaded in Phase 1B. The official Issue draft is docs/SPANUQ_ISSUE_DRAFT.md and requires user review before publication.

## Technical recommendation

Choose route B and proceed, only after user approval, with fixed Premise/Expected Consequence mean token NLL and a versioned Internal Signal Interface. See INTERNAL_SIGNAL_FALLBACK.md for meaning, raw-logit versus temperature-warped probabilities, cost, BF16 numerical caveat, real feasibility results and future SpanUQ replacement. Keep the validated hidden-state pipeline and strict SpanUQ loader intact. External Experience Memory, training-free Gate and paper-aligned benchmark plan are unchanged.
