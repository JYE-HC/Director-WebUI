# Director Web modifications to RayLight

This directory is based on RayLight 1.8.0 at upstream commit
`4085a37b62d7504cf7746f50c9d59e4fd1495b2b` and retains its Apache-2.0
license. It is a Director-maintained derivative and is not endorsed by the
upstream RayLight project.

Director-specific changes include:

- MiniMax H3 packed-layout, conditional-audio and distributed LoRA support;
- Comfy Kitchen INT8 attention for the supported Ulysses topology;
- logical GPU mapping under an existing `CUDA_VISIBLE_DEVICES` mask;
- cleanup limited to the selected Ray GPU pool;
- worker model retention and bounded host-RAM LRU caching;
- stable Ray actor lifecycle and shutdown behavior used by Director;
- private `directordeck_raylight` Python package and Ray-worker namespace, so
  an external user-installed `raylight` package can coexist without being read
  or replaced by DirectorDeck;
- normal logs omit the full CUDA visibility mask, which can contain GPU/MIG UUIDs.

Files changed from upstream carry a `Modified for Director Web` header. New
files carry an `Added for Director Web` header. The tests in this directory are
upstream/fork maintenance tests; ordinary users should rely on the backend
test suite and `tools/validate_native_comfy_prompts.py` instead.

Known limitation: the RayLight loader rejects a pruned/curve MiniMax H3 base
when a LoRA contains AdaLN weights. The bundled Standard Turbo loader supports
that specialized combination; the RayLight loader does not claim equivalence.
