# torch.compile: Dynamo, AOTAutograd, Inductor

All three are still the core layers of `torch.compile()` in current PyTorch; the
boundaries between them are unchanged from when they were first introduced.

## 1. TorchDynamo (`torch/_dynamo/`)

The frontend/graph-capture layer. Hooks CPython's PEP 523 frame-eval API
(`torch/csrc/dynamo/eval_frame.c`) to intercept a function's bytecode,
symbolically executes it (`symbolic_convert.py`'s `InstructionTranslator`),
and records the tensor ops into an FX graph while installing runtime
*guards* (assumptions that must hold for the compiled code to be reused).
Where it can't safely trace something, it takes a **graph break** and falls
back to eager for that part. Its output is one or more FX graphs handed to a
**backend**.

Detailed architecture doc: `torch/_dynamo/CLAUDE.md` (VariableTracker system,
guard tree, graph-break mechanics, C++ runtime).

## 2. AOTAutograd (`torch/_functorch/aot_autograd.py` + `torch/_functorch/_aot_autograd/`)

Turns a forward-only FX graph into a joint forward+backward graph *ahead of
time*, so both can be compiled together instead of relying on eager autograd
at runtime. Entry point is `aot_module_simplified`. Also handles
functionalization (removing mutations/views into pure ops) and decomposes
ops into a smaller canonical set.

## 3. TorchInductor (`torch/_inductor/`)

The backend/codegen layer. `compile_fx()` (`torch/_inductor/compile_fx.py:2889`)
is what Dynamo calls when you use `backend="inductor"` (see
`torch/_dynamo/backends/inductor.py`). It invokes AOTAutograd internally to
get the joint graph, then lowers it to generated Triton kernels (GPU) or
C++/OpenMP (CPU), doing fusion, scheduling, and memory planning along the way.

## Pipeline

```
torch.compile(fn) -> Dynamo captures FX graph -> calls backend("inductor")
  -> compile_fx() -> AOTAutograd (joint fwd/bwd graph) -> Inductor codegen (Triton/C++)
```

## Step-by-step: capture, compile, launch

File:line references below are from this checkout.

### Phase 1 — Capture (TorchDynamo)

1. **Wrapping.** `torch.compile(model)` wraps the callable in `OptimizedModule`
   (`torch/_dynamo/eval_frame.py:469`), which installs an
   `_TorchDynamoContext`. Nothing is traced yet.
2. **First call — frame interception.** CPython's PEP 523 hook (installed via
   `_PyInterpreterState_SetEvalFrameFunc` in `torch/csrc/dynamo/eval_frame.c`)
   intercepts the frame before it executes. `eval_frame_cpp.cpp`'s
   `dynamo__custom_eval_frame` checks the code object's `ExtraState` for a
   cached compiled version; on a cold call it calls back into Python:
   `ConvertFrameAssert.__call__` → `_compile()` (`convert_frame.py:1647`) →
   `transform()` (`convert_frame.py:1563`).
3. **Symbolic execution.** `symbolic_convert.py`'s `InstructionTranslator`
   walks the bytecode instruction-by-instruction, maintaining a symbolic
   `stack`/`symbolic_locals` of `VariableTracker`s. Tensor ops become FX graph
   nodes (owned by `OutputGraph`); assumptions relied on become **guards**.
   Anything unsafe to trace triggers a **graph break**: compile what's been
   captured so far, resume tracing after that point in a new frame.
4. **Finalizing the graph.** `OutputGraph.compile_subgraph()`
   (`output_graph.py:1992`) → `compile_and_call_fx_graph()`
   (`output_graph.py:2791`) builds the real `fx.GraphModule`, then
   `call_user_compiler()` (`output_graph.py:3159`) hands `(gm, example_inputs)`
   to the backend — by default `backend="inductor"`
   (`torch/_dynamo/backends/inductor.py`), which calls `compile_fx(*args, **kwargs)`.

### Phase 2 — Compile (AOTAutograd -> Inductor)

5. **Entry.** `compile_fx()` (`torch/_inductor/compile_fx.py:2889`) is
   "responsible for calling into AOT Autograd" — it doesn't compile the raw
   graph directly. It wraps `fw_compiler`/`bw_compiler`/`partition_fn`
   (Inductor-side) and calls `aot_autograd(...)` (`compile_fx.py:3275`), which
   calls `aot_module_simplified` in `torch/_functorch/aot_autograd.py`.
6. **Joint graph tracing.** AOTAutograd re-traces under fake tensors,
   capturing forward and backward together (the "joint graph"), so autograd
   doesn't run eagerly at runtime. It also **functionalizes** (mutations/views
   -> pure ops) and **decomposes** ops into a smaller canonical set.
7. **Partitioning.** A min-cut partitioner splits the joint graph back into a
   standalone forward graph and backward graph, choosing which activations to
   save vs. recompute.
8. **Handoff to Inductor per-subgraph.** Forward and backward graphs each go
   through `compile_fx_inner` (`compile_fx.py:857`) →
   `fx_codegen_and_compile` (`compile_fx.py:1927`) → (in-process mode)
   `_InProcessFxCompile.codegen_and_compile` (`compile_fx.py:1372`).
9. **Lowering.** `GraphLowering` (`torch/_inductor/graph.py:386`, an
   `fx.Interpreter` subclass) walks the FX graph via
   `graph.run(*example_inputs)` (`compile_fx.py:1648`), lowering each op from
   an ATen call into Inductor IR (buffers, layouts, scheduler nodes).
10. **Scheduling and codegen.** `Scheduler` (`torch/_inductor/scheduler.py:4210`)
    decides fusion groups, memory planning, and ordering; `scheduler.codegen()`
    (`graph.py:3002`) emits generated Triton kernel source (GPU) or C++/OpenMP
    (CPU) plus a Python wrapper module that allocates buffers and calls kernels
    in order.
11. **Kernel compilation.** Triton sources are compiled by `AsyncCompile`
    (`torch/_inductor/async_compile.py`) via a subprocess pool (parallel
    builds); the wrapper module loads through `PyCodeCache`. Result: an
    `OutputCode`/`CompiledFxGraph` — a real Python callable plus metadata
    (required input strides/alignment guards, CUDA graph eligibility).
12. **Wiring into autograd.** `AOTDispatchAutograd.build()`
    (`torch/_functorch/_aot_autograd/runtime_wrappers.py:3339`) wraps the
    compiled forward/backward callables into a real `torch.autograd.Function`
    subclass — `CompiledFunction` (`runtime_wrappers.py:3500`) with
    `CompiledFunctionBackward` nested inside — so eager `loss.backward()`
    correctly dispatches into the compiled backward graph later.
13. **Caching.** `AOTAutogradCache`
    (`torch/_functorch/_aot_autograd/autograd_cache.py:1010`) and Inductor's
    `FxGraphCache` key on a hash of graph + config and persist compiled
    artifacts to disk, so later processes can skip straight to a cached
    `OutputCode`.

### Phase 3 — Launch (runtime execution)

14. **Returning to Dynamo.** The compiled callable propagates back to
    `compile_subgraph`, which generates new bytecode (`codegen.py`'s
    `PyCodegen`) that loads graph inputs from their sources, calls the
    compiled callable, unpacks outputs, and replays tracked side effects.
15. **Installing the cache entry.** This bytecode plus the guard tree built
    during tracing is installed as a `CacheEntry` in the code object's
    `ExtraState` (`torch/csrc/dynamo/extra_state.cpp`) — an LRU list, since one
    function can have multiple compiled variants for different guard
    conditions.
16. **Every subsequent call.** The PEP 523 hook fires; `eval_frame_cpp.cpp`
    builds a `FrameLocalsMapping` and evaluates the guard tree — a C++
    `RootGuardManager` (`torch/csrc/dynamo/guards.cpp`) that fails fast. On a
    **guard hit**, cached bytecode runs directly via a shadow frame — no
    Python-level Dynamo/AOTAutograd/Inductor code runs, just the generated
    wrapper calling compiled kernels. On a **guard miss**, it tries other
    `CacheEntry`s in the LRU list, or recompiles (Phase 1 again, subject to a
    recompile limit before falling back to eager).
17. **Backward.** When the user calls `.backward()`, eager autograd sees the
    `CompiledFunction` node and invokes its compiled backward directly — no
    Python tracing overhead, since AOTAutograd already produced that graph in
    steps 6-7. (`torch._dynamo.compiled_autograd`, if enabled, can further
    capture/compile the backward call itself — an optional layer on top.)
18. **CUDA graphs (optional).** If eligible (static input addresses, no
    unsupported ops), the compiled callable's kernel-launch sequence can be
    captured once as a CUDA graph and replayed on later calls, skipping
    per-kernel launch overhead — transparent to the steps above.

## What's evolved on top (not a replacement of the stack)

- `torch/_dynamo/compiled_autograd.py` — compiling the backward pass itself,
  for cases where AOTAutograd's ahead-of-time backward isn't sufficient
  (e.g. dynamic control flow in backward).
- AOTAutogradCache / FxGraphCache — cross-run caching.
- `torch.export` — reuses Dynamo's capture but skips Inductor.

Dynamo is still the only supported frontend for `torch.compile`, AOTAutograd
is still the joint-graph layer, and Inductor is still the default/primary
backend.
