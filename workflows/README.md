# Example workflows

Drop these into `ComfyUI\user\default\workflows\`, then **Workflow → Open** in ComfyUI.

They are examples, not part of the patch. All three assume the [4-step Turbo LoRA](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo) is installed — remove the `MiniMaxH3TurboLoRA` node and swap `MiniMaxH3TurboSampler` for `KSamplerSelect` (with ~20 steps) if you don't want it.

| workflow | what it's for | canvas | time |
|---|---|---|---|
| `TURBO 4-step` | single shots | 864×480 / 124f (5.2 s) | ~85 s |
| `TURBO 4-step CHAIN` | continue a scene | 640×352 / 243f (10.1 s) | ~3 min |
| `TURBO REF2VA multi-character` | lock characters to reference images | 640×352 / 124f | ~50 s |

Each carries a MarkdownNote panel documenting its own settings.

## Model filenames

The loader nodes reference these names — rename or re-point them to match your files:

```
minimax_h3_fl2va_pruned_fp8_scaled.safetensors     diffusion_models/
minimax_h3_ref2va_pruned_fp8_scaled.safetensors    diffusion_models/  (REF2VA only)
qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors       text_encoders/
minimax_h3_video_vae_fp16.safetensors              vae/
minimax_h3_audio_vae_fp32.safetensors              vae/
minimax_h3_turbo_4step_ema_ckpt850.safetensors     loras/
```

## Two rules these workflows bake in

**Steps stays at 4.** The Turbo LoRA is step-*distilled* — raising it to 8 makes output worse, not better. See the main README.

**Keep `ref_image_size` on `match`** in the REF2VA workflow. The `max` option uses a 2048 short edge, and reference tokens ride through *every* sampling step, so it is several times slower.

## Character consistency

**Continuing a scene** → the CHAIN workflow. Its `first_frame` / `last_frame` LoadImage nodes ship **muted**; select one and press **Ctrl+M** to enable. Get the frame with:

```bash
python extract_last_frame.py myclip.mp4
```

That writes into `ComfyUI\input\` — `LoadImage` cannot read from `output\`.

**Introducing or pairing characters** → the REF2VA workflow. Up to 9 reference images, addressed in the prompt in slot order as `<Picture 1>`, `<Picture 2>`, `<Picture 3>`. `ref_image_0` is `<Picture 1>`.

Also keep a **verbatim character description** in every prompt — the same sentence, word for word. Paraphrasing is how designs drift between clips.
