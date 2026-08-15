"""RDNA4 (gfx12xx) speed patches for MiniMax-H3 in ComfyUI.

Two edits to comfy/ldm/minimax/model.py. Both are defensively gated: if the
required backend is missing they return to stock behaviour rather than failing,
so the patched file is safe on any GPU.

PATCH 1 -- partial-rope HIP fast path
    MiniMax-H3 uses *partial* rotary: rot_dim 96 inside a 128-wide head.
    comfy-kitchen's fused RMSNorm+rope WMMA kernel only accepts
    rot_dim == head_dim and hands anything else to a pure-PyTorch path, so all
    52 attention blocks silently lose their kernel. Splitting the op --
    full-width RMSNorm, then the HIP rope on the rotated prefix -- restores it.

    Measured on a Radeon AI PRO R9700 (gfx1201), BIT-EXACT vs stock
    (max_abs_diff = 0.0 on identical weights and inputs):
        rope op alone    1.97-2.04x   (9.34 ms -> 4.74 ms at L=8192)
        full attn block  1.07x @ L=8192, 1.18x @ L=2048

PATCH 2 -- autotuned Triton attention
    Routes H3's long-sequence attention through flex_attention compiled with
    max-autotune instead of AOTriton SDPA. On gfx1201 Inductor selects
    BLOCK_M=128 / BLOCK_N=16 / num_warps=4; BLOCK_N=16 matches RDNA4's WMMA
    width and AOTriton ships no equivalent config.

    Measured at H3's real shape (B=1, H=56, L=15488, D=128, bf16):
        SDPA (AOTriton flash)         141.7 ms   48.6 TFLOPS
        flex_attention, defaults      120.7 ms   57.0 TFLOPS   1.17x
        flex_attention, max-autotune   94.1 ms   73.1 TFLOPS   1.51x

    NOT bit-exact: 2.4e-4 per call (bf16 rounding, different accumulation
    order -- not reduced precision). Diffusion amplifies this, so a full
    20-step render lands ~23.5 dB PSNR against the SDPA render: a different
    sample from the same distribution, equivalent quality, NOT the same frames.
    Set H3_FLEX_ATTENTION=off to reproduce stock output exactly.

Usage
    python apply_h3_rdna4_patches.py                 # apply (auto-detect ComfyUI)
    python apply_h3_rdna4_patches.py --comfy-path X  # point at a ComfyUI checkout
    python apply_h3_rdna4_patches.py --revert        # restore stock file
    python apply_h3_rdna4_patches.py --check         # report status only

Idempotent: running twice is a no-op. A .orig-backup is written before the
first edit. Every anchor must match exactly once or the script refuses, so it
will not apply a stale patch to a changed upstream file.
"""
import argparse
import pathlib
import shutil
import sys

MARKER = "_hip_partial_rope_available"

ANCHOR_IMPORT = "import math\n\nimport torch\nimport torch.nn as nn\n"
NEW_IMPORT = ("import functools\nimport logging\nimport math\nimport os\n\n"
              "import torch\nimport torch.nn as nn\n")

ANCHOR_HELPERS = "def rope_rotation_table(angles, dtype):"

HELPERS = '''@functools.lru_cache(maxsize=1)
def _hip_partial_rope_available():
    """True when comfy-kitchen's HIP backend can serve rope on this GPU."""
    try:
        from comfy_kitchen.backends import hip
    except Exception:
        return False
    try:
        return bool(hip.is_available() and hip.has_wmma())
    except Exception:
        return False


def _rms_rope_partial_hip(q, k, rope_freqs, qw, kw, eps, rot):
    """RMSNorm over the full head, HIP split-half rope over the first `rot` dims.

    Equivalent to eager's rms_rope_split_half_ at this rot_dim: that path
    rms_norms the whole head, rotates [..., :rot] and concatenates the
    untouched tail. Here the tail is never written, so the concat and its
    allocation disappear too.
    """
    from comfy_kitchen.backends import hip

    head_dim = q.shape[-1]
    q = nn.functional.rms_norm(q, (head_dim,), weight=qw, eps=eps)
    k = nn.functional.rms_norm(k, (head_dim,), weight=kw, eps=eps)
    hip.apply_rope_split_half_(q[..., :rot], k[..., :rot], rope_freqs)
    return q, k


# Below this packed-sequence length, stay on SDPA: the gain is negligible and
# Inductor autotunes a fresh kernel per distinct shape.
FLEX_MIN_SEQ = 4096


def _flex_attention_mode():
    """'auto' (default), 'off', or 'force' -- from H3_FLEX_ATTENTION."""
    return os.environ.get("H3_FLEX_ATTENTION", "auto").strip().lower()


@functools.lru_cache(maxsize=1)
def _flex_attention_fn():
    """An autotuned Triton attention kernel, or None to stay on SDPA.

    Autotuning costs ~40 s the first time a new sequence length is seen; set
    TORCHINDUCTOR_CACHE_DIR so that cost is paid once per machine, not per run.
    """
    if _flex_attention_mode() == "off":
        return None
    try:
        import torch.nn.attention.flex_attention as fa

        compiled = torch.compile(fa.flex_attention, dynamic=False,
                                 mode="max-autotune-no-cudagraphs")
    except Exception as e:
        logging.info("MiniMax-H3: flex_attention unavailable (%s); using SDPA.", e)
        return None
    logging.info("MiniMax-H3: using autotuned flex_attention "
                 "(H3_FLEX_ATTENTION=off to disable).")
    return compiled


def rope_rotation_table(angles, dtype):'''

ANCHOR_ROPE = """            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)"""

NEW_ROPE = """            elif rot != self.head_dim and _hip_partial_rope_available():
                # partial rotary would fall off the HIP kernel onto eager; split
                # the op so the rope half keeps its WMMA kernel
                q, k = _rms_rope_partial_hip(q, k, rope_freqs, qw, kw, self.q_norm.eps, rot)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)"""

ANCHOR_ATTN = ("        out = optimized_attention(q, k, v, self.heads, mask=None, "
               "skip_reshape=True, transformer_options=transformer_options)\n"
               "        return self.out_proj(out.squeeze(0))")

NEW_ATTN = """        out = None
        # Only worth it on the long packed sequence. The 2 token-refiner layers
        # run a few hundred text tokens, where the win is nil and autotuning a
        # separate kernel per shape costs more than it saves.
        def _unwrap(t):
            # torch.Tensor on ComfyUI < v0.32; AttentionTensorContainer on
            # v0.32+ (comfy/ldm/modules/attention.py). .peek() reads without
            # consuming, so the container stays valid for the
            # optimized_attention() fallback below if flex_attention raises.
            return t.peek() if hasattr(t, "peek") else t
        flex = (_flex_attention_fn()
                if _unwrap(q).shape[2] >= FLEX_MIN_SEQ else None)
        if flex is not None:
            try:
                # no .contiguous(): q/k/v are strided views of the packed qkv
                # buffer and copying them is expensive. Inductor handles strides.
                o = flex(_unwrap(q), _unwrap(k), _unwrap(v))
                # flex returns [1, heads, seq, head_dim]; optimized_attention
                # hands back [1, seq, heads*head_dim] already flattened.
                out = o.transpose(1, 2).reshape(1, o.shape[2], self.heads * self.head_dim)
            except Exception as e:
                if _flex_attention_mode() == "force":
                    raise
                logging.warning("MiniMax-H3: flex_attention failed (%s); "
                                "falling back to SDPA.", e)
                _flex_attention_fn.cache_clear()
                os.environ["H3_FLEX_ATTENTION"] = "off"
                out = None
        if out is None:
            out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
        return self.out_proj(out.squeeze(0))"""

EDITS = [("import block", ANCHOR_IMPORT, NEW_IMPORT),
         ("rope_rotation_table", ANCHOR_HELPERS, HELPERS),
         ("rope dispatch", ANCHOR_ROPE, NEW_ROPE),
         ("attention call", ANCHOR_ATTN, NEW_ATTN)]


def find_comfy(explicit):
    """Locate a ComfyUI checkout containing the MiniMax-H3 model file."""
    here = pathlib.Path(__file__).resolve().parent
    cands = []
    if explicit:
        cands.append(pathlib.Path(explicit).expanduser().resolve())
    cands += [here, here / "ComfyUI", here.parent, here.parent / "ComfyUI",
              here.parent.parent / "ComfyUI"]
    for c in cands:
        if (c / "comfy" / "ldm" / "minimax" / "model.py").exists():
            return c
    sys.exit(
        "Could not find ComfyUI.\n"
        "Looked for comfy/ldm/minimax/model.py under:\n  " +
        "\n  ".join(str(c) for c in cands) +
        "\n\nPass it explicitly:\n"
        "  python apply_h3_rdna4_patches.py --comfy-path C:\\path\\to\\ComfyUI\n\n"
        "If that path exists but has no comfy/ldm/minimax/, your ComfyUI predates\n"
        "MiniMax-H3 support (needs v0.30.0 or newer)."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy-path", help="path to the ComfyUI folder")
    ap.add_argument("--revert", action="store_true", help="restore the stock file")
    ap.add_argument("--check", action="store_true", help="report status, change nothing")
    args = ap.parse_args()

    comfy = find_comfy(args.comfy_path)
    target = comfy / "comfy" / "ldm" / "minimax" / "model.py"
    backup = target.with_suffix(".py.orig-backup")
    src = target.read_text(encoding="utf-8")
    patched = MARKER in src

    print(f"ComfyUI : {comfy}")
    print(f"target  : {target}")
    print(f"status  : {'PATCHED' if patched else 'stock'}")

    if args.check:
        return

    if args.revert:
        if not backup.exists():
            sys.exit("no .orig-backup found - nothing to revert to")
        shutil.copyfile(backup, target)
        print("reverted to stock.")
        return

    if patched:
        print("already patched - nothing to do.")
        return

    for name, anchor, _ in EDITS:
        n = src.count(anchor)
        if n != 1:
            sys.exit(
                f"\nAnchor '{name}' found {n} times (expected exactly 1).\n"
                f"This ComfyUI's model.py differs from the version these patches\n"
                f"were written against (v0.30.0). Refusing to apply a stale patch.\n"
                f"Re-derive the edits by hand - see the docstring at the top."
            )

    if not backup.exists():
        shutil.copyfile(target, backup)
        print(f"backup  : {backup}")

    for _, anchor, new in EDITS:
        src = src.replace(anchor, new, 1)
    target.write_text(src, encoding="utf-8")

    print("\nPATCHED.")
    print("  + partial-rope HIP fast path (bit-exact)")
    print("  + autotuned Triton attention (H3_FLEX_ATTENTION=off to disable)")
    print("\nFully restart ComfyUI - a browser refresh does not reload Python.")


if __name__ == "__main__":
    main()
