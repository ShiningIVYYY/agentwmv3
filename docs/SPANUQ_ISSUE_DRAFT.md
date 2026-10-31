# DRAFT — user review required; not posted

Suggested title: Qwen3-4B released checkpoint cannot strictly load: missing u_conditioner implementation and PE extent mismatch

Hello, thank you for releasing SpanUQ. I am trying to run the published Qwen3-4B probe with the official frozen inference code, without training or discarding checkpoint tensors. Could you point me to the matching implementation/export path?

## Versions and assets

- Source commit: 778b1c1e3eeeda66233fe6b426f9303f4c7f76a3 (all four currently advertised commits share the model/config/producer blobs).
- HF DamonDemon/SpanUQ revision: 97ca9c820ac3561259956ec21cbdd984c3132d37, subfolder Qwen3-4B.
- best_model.pt SHA256: 78a203415037cb6645385d27b14d4207d184d522cc2c33af07098f2f2254fb12 (same LFS object since original upload 68053f75).
- Backbone: Qwen/Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c.
- Python 3.12.0; torch 2.8.0+cu128; Transformers 4.57.3; tokenizers 0.22.2. Strict-loading reproduction below is CPU-only.
- Actual backbone/decoder-block extraction works; this issue concerns probe compatibility.

## Minimal reproduction

Clone/check out the source commit above; use a compatible isolated environment with torch and huggingface_hub. From the source root:

~~~python
import json
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download
from spanuq.config import SpanUQConfig
from spanuq.model import SpanUQ_DETR

revision = "97ca9c820ac3561259956ec21cbdd984c3132d37"
paths = {
    name: Path(hf_hub_download(
        "DamonDemon/SpanUQ", f"Qwen3-4B/{name}", revision=revision))
    for name in ("model_config.json", "best_model.pt")
}
raw = json.loads(paths["model_config.json"].read_text())
try:
    SpanUQConfig(**raw)
except TypeError as error:
    print("Unmodified config:", error)

# Explicit legacy-field translation for deeper diagnosis only.
# No checkpoint state key/value is removed or changed.
values = dict(raw)
values["mob_n_components"] = values.pop("mob_k")
values["seq_aggregation"] = values.pop("seq_agg_mode")
response_limit_metadata = values.pop("max_seq_len")
print("Published input-limit metadata:", response_limit_metadata)
config = SpanUQConfig(**values)
model = SpanUQ_DETR(**config.model_kwargs())
state = torch.load(paths["best_model.pt"], map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
~~~

The unmodified constructor rejects mob_k. After the explicitly recorded aliases, strict loading reports four unexpected keys:

~~~text
u_conditioner.cond_mlp.0.weight [256,3]
u_conditioner.cond_mlp.0.bias   [256]
u_conditioner.cond_mlp.2.weight [512,256]
u_conditioner.cond_mlp.2.bias   [512]

size mismatch for token_encoder.pos_enc.pe:
checkpoint [1,16384,512] versus model [1,4096,512]
~~~

There are 134 checkpoint tensors versus 130 in the instantiated public model: no missing public-model keys, 129 common shape matches and the one PE shape mismatch. The 16384 buffer exactly matches the official sinusoidal formula generated independently at that length, including the current 4096 prefix (maximum error 0). I have not cropped/replaced it or patched the constructor to bypass the missing module.

## Questions

1. Could you release the exact model class/forward used for this checkpoint, including the three conditioner inputs, module operations and where its output enters inference? If it is training-only, is there an official validated export/inference path for this exact checkpoint?
2. What constructor/config field sets the PE maximum length to 16384? How should mob_k, seq_agg_mode and max_seq_len map to the released classes?
3. Is seq_aggregation intended to control model behavior? In the inspected code, model_kwargs does not pass it and forward uses importance aggregation directly.
4. Can you confirm the training backbone/tokenizer revision and layer convention? The batch hidden-state producer/demo hook decoder blocks [21,22,23], equivalent to HF hidden_states [22,23,24], whereas the on-the-fly evaluator indexes hidden_states [21,22,23].
5. The demo also passes token_entropies to forward and reads pred_start/pred_end, while the model signature/output use a different interface. Is a matching end-to-end inference script available?

I have not used strict=False, dropped weights, guessed an unused conditioner or retrained the probe. I would appreciate a matching commit/config and strict-load inference example. Thank you.
