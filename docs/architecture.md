# Director Web architecture

## Product boundary

Director Web edits one long-film timeline. Each segment chooses only the FL2VA
or Ref2VA model family; T2V, I2V, FL2V, R2V, V2V and RV2V are server-derived
recipes, not six isolated workspaces. The canonical timeline is the only
authoring document; ComfyUI graphs are ephemeral execution artifacts.

```text
Browser (typed timeline/settings only)
                |
                | REST + task polling
                v
Director API (FastAPI + SQLite)
    | validation / fixed graph compiler / parent-child jobs / ffmpeg assembly
    |
    | standard HTTP /prompt + /queue + /history + /view
    | standard WebSocket progress + bounded metadata preview
    v
ComfyUI core + comfy_extras
    + server-derived H3 LoRA loader when a LoRA is selected
    + GPU-pool-derived RayLight backend for multi-GPU execution
```

The browser never receives or submits an executable prompt. `/api/timeline/compile`
returns only a redacted plan, model families, resolved execution backends and
node-policy metadata. The job endpoint always recompiles from validated data.

## Canonical domains

SQLite stores:

- one shared runtime settings document;
- one unified ordered timeline document;
- uploaded asset identity, canonical ComfyUI origin, content hash and probe facts;
- parent jobs and immutable configuration/settings/compiler manifests;
- one child workflow unit per selected segment, with prompt ID, progress,
  sampler/output-node ownership and exact outputs.

Timeline document v2 gives each segment stable identity, order, enabled state,
duration, prompt and a strict family-discriminated payload. Switching between
FL2VA and Ref2VA rebuilds that payload from an allowlist; hidden fields from the
other family cannot survive validation. Compile plans keep `mode` as that
family and expose the derived six-way value separately as `recipe`.

Shared settings comprise the ComfyUI endpoint/client identity, FL2VA and
Ref2VA model bindings, optional family LoRA, CLIP, video/audio VAE and requested
logical GPU pools/devices. The pool size is the sole Standard/RayLight route. Segment prompts,
project render settings, per-family sampling, references and source trims belong
to the unified timeline.

Recipe inference is deterministic and fixed:

| Family | Typed media | Recipe |
| --- | --- | --- |
| FL2VA | no anchors / first only / any last | T2V / I2V / FL2V |
| Ref2VA | references only / source only / source + any reference | R2V / V2V / RV2V |

Ref2VA deliberately keeps `source_video` separate from independent
`reference_videos`. Source occupies the first official reference-video input;
the stock node independently supports nine reference images, three video
channels and three standalone reference audios. A source may therefore be
combined with up to two independent videos without reducing either the image
or standalone-audio capacity. Its optional paired soundtrack uses the matching
video-audio input rather than a standalone-audio slot. An empty Ref2VA segment
may be saved as unfinished editing state but fails compile/submit closed.

Both diffusion selectors receive the complete ComfyUI `diffusion_models`
inventory. Filenames are never used to classify a model family.

## Server-owned execution

The backend emits one ComfyUI prompt unit for every selected segment. Standard
units run before RayLight units, then by family and timeline order. Prompts in
the same family/backend repeat stable loader node ids and inputs so ComfyUI can
reuse its endpoint-local cache, without making all segments share one graph,
failure domain, cancellation boundary or decode cache. RayLight units remain
serialized because initialization owns process-global Ray state. Serialization
is terminal-gated: only one Director Ray generation is visible in ComfyUI at a
time, and its exact successful history unlocks the next same-parent or later-parent
unit. Failure or external removal taints the actor pool and requires RayKill plus
a fresh epoch before any remaining generation. The create API returns a durable
`preparing` snapshot after preflight and endpoint-ticket registration; the managed
dispatcher continues independently.

The standard graph uses only core/official nodes:

```text
UNETLoader -> SelectModelDevice -> optional allowed LoRA -> MiniMaxH3SigmaShift
CLIPLoader -> SelectCLIPDevice
VAELoader  -> SelectVAEDevice

MiniMaxH3ImageToVideo | MiniMaxH3ReferenceToVideo
  -> BasicGuider + BasicScheduler + KSamplerSelect + RandomNoise
  -> SamplerCustomAdvanced
  -> VAEDecode + VAEDecodeAudio
  -> CreateVideo -> SaveVideo
```

RayLight keeps official conditioning, media loading, VAE decode and output,
but replaces the model/sampler chain with the exact RayLight allowlist. It is
not connected to or disguised as a standard `MODEL` graph. The
`MiniMaxH3Director` custom node is forbidden in all executable templates.

Native H3 v1 fixes render FPS at 24 and uses `BasicGuider`, so the product has
no inference CFG field or negative prompt. Video assets are normalized by the backend to
immutable 24fps proxies before upload. Autogrow reference inputs require dense
per-modality slots `0..N-1`; the backend rejects sparse bindings rather than
silently renumbering prompt labels.

## Parent/child lifecycle

A timeline submission first persists one parent and its child unit rows, then:

1. validates assets against their stored origin and metadata;
2. compiles fixed graphs and asserts every node against the selected policy;
3. checks capabilities, inventory, logical devices and Ray topology;
4. submits child prompts in deterministic order;
5. reconciles child lifecycle through standard queue/history;
6. consumes standard WebSocket prompt/node lifecycle events for loading,
   conditioning, sampling, decode and save stages; when a sampler reports
   main-process progress, node IDs additionally map to exact `step/total`
   (the installed RayLight worker normally exposes only its sampling stage);
7. requires exactly one expected output for every selected segment;
8. for a full export, downloads exact child outputs, normalizes/concatenates
   them with ffmpeg and registers one final ComfyUI output.

Cancellation, progress, final assembly and deletion use compare-and-set state
transitions. A late queue response, WebSocket frame or assembly completion
cannot revive a cancelled/terminal task. Losing the WebSocket only removes
live stage/step detail; queue/history remain the durable lifecycle authority. Each
background reconciliation pass uses one queue snapshot and one bounded
bulk-history read. When queue is
available, at most 16 missing prompt IDs are checked exactly and the window
rotates across later polls; when queue is unavailable, bulk positives may
advance children but absence is never treated as cancellation.

Whole-timeline ffmpeg assembly is owned only by the process reconciler. HTTP
job list/detail reads are SQLite-only for both parent/child and historical
legacy rows; a disconnected or black-hole ComfyUI endpoint therefore cannot
freeze the task panel. Segment-take proxy reads resolve the persisted child
output map directly. Cancellation is the sole HTTP operation allowed to
preempt an assembly flight.

On restart, lifespan performs only a local transaction before yielding: dead
submission ownership becomes a restart-recovery marker. A separate bounded,
cancellable worker then targets every caller-assigned prompt ID through the
atomic cancel API. An unconfirmed cancel remains recovery-owned and is retried;
temporary queue/history absence cannot close it because an old `/prompt`
handler may still enqueue after validation. Every newer Director dispatcher on
that endpoint waits behind all bound recovery-owned children—including old
Standard prompts—so a delayed enqueue cannot cross a new Ray sequence. Shutdown cancels and gathers both
the reconciler and recovery worker.

The public job response exposes `segment_results` only when one stable segment
maps unambiguously through its child-owned SaveVideo node to one output. This
lets the program monitor show the newest candidate take without exposing raw
workflow JSON. Full-timeline assembly still has its own parent output and fails
closed on missing, duplicate or unknown segment outputs.

Metadata-bearing ComfyUI event-4 PNG/JPEG previews are accepted only for a
registered active child sampler and are capped at 2 MiB. The latest frame is an
in-memory TTL/LRU cache entry, never a database blob. Its same-origin endpoint
is advertised only while still valid and responds with `no-store` and
`nosniff`; terminal/deleted jobs cannot be revived by a late frame.

## GPU and residency

`gpu:N` is a ComfyUI-process logical index and may be remapped by
`CUDA_VISIBLE_DEVICES`; it is never presented as a physical GPU ID.

Per model family:

- a one-GPU pool resolves to Standard, using the saved model device and official selector nodes;
- a pool of two or more resolves to RayLight and requires a topology whose product
  exactly equals the pool size;
- legacy `backend=standard|raylight` values are accepted only for migration,
  canonicalized to `auto`, and ignored by the compiler;
- a derived RayLight graph that cannot be honored fails before queueing—there is no
  silent standard fallback.

The standard compiler never adds an unload/free-memory node, so stable loader
inputs can reuse ComfyUI's model cache. RayLight defaults to
`keep_until_switch`: Director retains one exact
family/model/LoRA/GPU/topology/loader/mutating-sigma key and compatible prompts reuse its
persistent epoch and CUDA weights. An incompatible Ray key or Standard prompt
first crosses a persisted `RayKill` barrier; only a positive history result
permits the next prompt, whose epoch is incremented so ComfyUI cannot reuse
actor handles from an already-shut-down pool. This permits FL2VA and Ref2VA to
run sequentially on the same GPUs without relying on OOM eviction or requiring
a ComfyUI restart. Every Director Ray initializer explicitly sets
`driver_cleanup_policy=ray_devices`. Before sampling, RayLight therefore frees
Comfy-managed driver models only on the local logical devices named by
`GPU_SELECT`; CLIP/VAE placed on non-Ray devices are not force-unloaded by this
path. They remain ordinary ComfyUI caches and may still be evicted under memory
pressure, model switching, an explicit Free Memory action, or restart, so this
is best-effort warmth rather than an absolute residency guarantee.

The explicit `release_after_sampling` option sets
`clear_vram_after_sampling=true`; it unloads worker weights after each sample
and applies the same `ray_devices` driver cleanup scope, but does not destroy
the Ray cluster. This scoped post-sample release safely makes shared Ray/auxiliary
devices available for VAE decode or a later workload. Director keeps tracking
the cluster and retains its epoch for the same key. The installed worker keeps
its ModelPatcher and active request key while unpatching/offloading CUDA weights,
so a cached loader output remains valid and the next sampler moves the same
model back to CUDA. With `keep_until_switch`, worker weights stay resident;
cross-process memory on an overlapping CLIP/VAE device cannot be coordinated by
ComfyUI alone, so simultaneous auxiliary residency is not guaranteed there. A
following Standard prompt still crosses the full loader-chain-to-`RayKill`
barrier. RayLight upgrades must re-audit this worker contract.

The persisted ledger records the exact loader subgraph, monotonic epoch, queue
tail prompt id, and a taint bit. Failed, cancelled, missing, or ambiguous tails
are never treated as reusable; switching first replays the typed
initializer -> optional LoRA -> UNET loader chain and connects its
`RAY_ACTORS` output to a unique `RayKill`, then requires exact successful
history. Internal barrier prompts are durable recovery/cancellation controls
but carry no timeline segment and do not affect public segment progress.
Pre-release v1 ledger rows do not contain a verifiable full loader chain.
Migration retains their embedded epoch but marks the old actor pool unknown;
Standard submissions fail closed until one Director RayLight task initializes a
new epoch (whose initializer explicitly shuts down the old local Ray pool) and
replaces the ledger with a complete v2 descriptor.
These guarantees cover only prompts serialized by Director. Manually submitted
ComfyUI workflows bypass its endpoint lock and runtime ledger; their
interference is unsupported and is not guaranteed to be detected, so they must
not be mixed into a Director-managed Ray residency sequence.

During upgrade, former family-pinned residency values are mapped once to
`keep_until_switch`. An explicitly saved `release_after_sampling` selection is
preserved, and a durable migration marker prevents later restarts from
rewriting that choice. Immutable historical job snapshots continue to record
the policy that actually compiled those jobs.

## Media and deletion boundary

Generated media remains owned by the captured ComfyUI instance. Director stores
only output references. Removing a ComfyUI history row or file does not erase
the Director audit job; deleting a terminal Director job removes only the local
record and deliberately preserves upstream files.

Assets are keyed by server-issued ID and bound to their upload origin. Client
paths are never trusted when compiling a graph. Switching endpoint makes old
assets unavailable for new jobs until they are uploaded to the new instance.
Deleting a referenced asset fails with 409 by default. Explicit `cascade=true`
runs one `BEGIN IMMEDIATE` transaction that unbinds the unified timeline and all
six legacy drafts, compacts local reference slots, rewrites affected prompt
tags, preserves the selected FL2VA/Ref2VA family while recipe inference follows
the remaining media, unbinds typed fields in
legacy mode drafts, revalidates every document and finally removes only the
local asset record. Any error rolls the whole transaction back; remote inputs
and generated outputs are always preserved. The response returns audit
`unbound_usages` and `outputs_preserved=true`; the browser then reloads the
authoritative timeline instead of reconstructing server decisions.

Existing timeline v1 documents cross a strict six-mode migration boundary:
legacy recipe objects are validated before conversion to the corresponding v2
family shape, so impossible old cross-mode fields do not become meaningful.
Live drafts normalize to v2; historical job snapshots and the six legacy draft
endpoints remain readable and are not reinterpreted as new execution audits.

## Safety checks

Submission fails closed when any of these contracts is missing:

- fixed node class allowlist and expected core/extras/custom provenance;
- known node provenance before submit, followed by ComfyUI's native prompt
  validator for exact input/output compatibility;
- selected model, LoRA and VAE/CLIP inventory entries;
- valid logical devices and RayLight topology;
- required typed assets, content identity, 24fps video proxy and source bounds;
- dense local reference slots and resolvable prompt tags;
- H3 frame/canvas limits, fixed FPS, absent inference CFG and supported continuity policy;
- graph links to existing nodes and server-owned SaveVideo prefixes.

The application is a single trusted-user local tool without authentication.
Network deployment requires an authenticated TLS reverse proxy and origin
restrictions.
