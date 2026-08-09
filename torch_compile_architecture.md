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

## What's evolved on top (not a replacement of the stack)

- `torch/_dynamo/compiled_autograd.py` — compiling the backward pass itself,
  for cases where AOTAutograd's ahead-of-time backward isn't sufficient
  (e.g. dynamic control flow in backward).
- AOTAutogradCache / FxGraphCache — cross-run caching.
- `torch.export` — reuses Dynamo's capture but skips Inductor.

Dynamo is still the only supported frontend for `torch.compile`, AOTAutograd
is still the joint-graph layer, and Inductor is still the default/primary
backend.
