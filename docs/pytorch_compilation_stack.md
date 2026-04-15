# PyTorch Compilation Stack: Dynamo → AOT → Inductor

This document provides a detailed architecture overview of PyTorch's compilation stack, including TorchDynamo, AOT Autograd, and Inductor.

## Overview

PyTorch 2.0 introduced a new compilation stack that transforms Python code into optimized machine code through multiple stages. This architecture enables significant performance improvements while maintaining Python's flexibility.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          User Python Code                                    │
│                       def forward(x, y):                                    │
│                           return torch.sin(x) + y                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ @torch.compile() or torch.compile()
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TorchDynamo: Graph Capture                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Frame Evaluation Hook                                            │   │
│  │    (sys.setprofile)                                                 │   │
│  │                                                                     │   │
│  │ 2. Bytecode Analysis                                                │   │
│  │    (CPython bytecode → FX Graph)                                   │   │
│  │                                                                     │   │
│  │ 3. Guard Generation                                                 │   │
│  │    (Shape checks, dtype checks, etc.)                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Output: torch.fx.GraphModule with guards                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ Graph Module
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AOT (Ahead-of-Time) Compiler                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input: FX GraphModule (Forward)                                     │   │
│  │                                                                     │   │
│  │ 1. Partitioning                                                     │   │
│  │    ┌────────────────────┐          ┌────────────────────┐          │   │
│  │    │  Forward Graph     │          │  Backward Graph    │          │   │
│  │    │  (Compiled)        │          │  (Compiled)        │          │   │
│  │    └────────────────────┘          └────────────────────┘          │   │
│  │           ▲                                ▲                        │   │
│  │           │                                │                        │   │
│  │  2. Autograd Analysis with torch.autograd.Function                   │   │
│  │     (Using tools/autograd/)                                         │   │
│  │     derivatives.yaml → Gradient Formulas                            │   │
│  │                                                                     │   │
│  │  3. Separate Compilation Units                                      │   │
│  │     forward.so + backward.so                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Output: AOT Compiled Artifacts (forward + backward)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ AOT Artifacts
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Inductor: Code Generation                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. FX Graph Lowering                                                │   │
│  │    (FX nodes → Inductor IR)                                        │   │
│  │                                                                     │   │
│  │    Inductor IR:                                                     │   │
│  │    - Computations: ir.Computation, ir.Buffer                       │   │
│  │    - Loops: ir.Loop                                                │   │
│  │    - Operations: ir.Pointwise, ir.Reduction, ir.Convolution        │   │
│  │                                                                     │   │
│  │ 2. Scheduling & Optimization                                        │   │
│  │    ┌──────────────────────────────────────────────────────────┐    │   │
│  │    │ • Loop fusion/tiling                                       │    │   │
│  │    │ • Memory planning (buffer reuse)                           │    │   │
│  │    │ • Operator fusion                                          │    │   │
│  │    │ • Tiling strategy                                          │    │   │
│  │    └──────────────────────────────────────────────────────────┘    │   │
│  │                                                                     │   │
│  │ 3. Code Generation Backends                                         │   │
│  │    ┌────────────────────┐          ┌────────────────────┐          │   │
│  │    │  Triton CodeGen    │          │  C++/CUDA CodeGen  │          │   │
│  │    │  (GPU Kernels)     │          │  (Fallback/CPU)    │          │   │
│  │    └────────────────────┘          └────────────────────┘          │   │
│  │                                                                     │   │
│  │ 4. Compilation & Caching                                            │   │
│  │    - Triton JIT compilation                                         │   │
│  │    - Kernel caching (inductor_codegen_cache)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Output: Optimized kernels (Triton .py files → .ttir → PTX)               │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ Optimized kernels
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Runtime Execution                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Guard Evaluation (from Dynamo)                                   │   │
│  │                                                                     │   │
│  │ 2. Dispatch to Compiled Kernels                                     │   │
│  │    if guards_pass:                                                  │   │
│  │        compiled_kernel(*args)                                      │   │
│  │    else:                                                           │   │
│  │        recompile() or fallback to eager                            │   │
│  │                                                                     │   │
│  │ 3. Cache Management                                                 │   │
│  │    - CompiledKernelCache                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Simplified Compilation Flow

```
                      ┌─────────────────────────┐
                      │   User Python Code      │
                      └───────────┬─────────────┘
                                  │
                                  │ @torch.compile()
                                  ▼
                      ┌─────────────────────────┐
                      │    TorchDynamo          │
                      │  (Graph Capture)        │
                      └───────────┬─────────────┘
                                  │ FX GraphModule
                                  ▼
                      ┌─────────────────────────┐
                      │    AOT Autograd         │
                      │  (Forward + Backward)   │
                      └───────────┬─────────────┘
                                  │ Forward & Backward Graphs
                                  ▼
                      ┌─────────────────────────┐
                      │       Inductor          │
                      │ (Code Generation)       │
                      └───────────┬─────────────┘
                                  │ Triton/CPP Kernels
                                  ▼
                      ┌─────────────────────────┐
                      │   Runtime + Cache       │
                      │ (Guarded Dispatch)      │
                      └─────────────────────────┘
```

## Key Components

### TorchDynamo

**Purpose**: Captures Python functions into FX graphs for compilation.

**Key Functions**:
- Intercepts Python bytecode execution via frame evaluation hooks
- Analyzes bytecode to construct FX graphs representing tensor operations
- Generates guards to ensure compiled code matches runtime conditions

**Important Files**:
- `torch/_dynamo/convert_frame.py` - Frame evaluation and graph capture
- `torch/_dynamo/symbolic_convert.py` - Bytecode analysis and graph construction
- `torch/_dynamo/guards.py` - Guard generation and verification

### AOT (Ahead-of-Time) Compiler

**Purpose**: Separates forward and backward graphs for training workloads.

**Key Functions**:
- Analyzes forward graph to generate corresponding backward graph
- Uses PyTorch's autograd system and derivatives.yaml for gradient formulas
- Creates separate compilation units for forward and backward passes

**Important Files**:
- `torch/_functorch/aot_autograd.py` - Main AOT compilation entry point
- `torch/_functorch/partitioners.py` - Graph partitioning logic

### Inductor

**Purpose**: Generates optimized machine code from FX graphs.

**Key Functions**:
- Lowers FX graphs to Inductor IR (intermediate representation)
- Applies optimizations like loop fusion, operator fusion, and memory planning
- Generates Triton code for GPUs and C++/CUDA code for CPUs/fallbacks

**Important Files**:
- `torch/_inductor/graph.py` - FX graph lowering to Inductor IR
- `torch/_inductor/scheduler.py` - Optimization scheduling
- `torch/_inductor/codegen/triton.py` - Triton code generation
- `torch/_inductor/codegen/cpp.py` - C++/CPU code generation

## Configuration

### Dynamo Configuration
```python
import torch._dynamo.config

# Key settings:
# - verbose: Enable detailed logging
# - suppress_errors: Continue on compilation failures
# - cache_size_limit: Maximum compiled functions to cache
```

### Inductor Configuration
```python
import torch._inductor.config

# Key settings:
# - debug: Enable debug output
# - max_fusion_size: Maximum number of nodes to fuse
# - triton.cudagraphs: Enable CUDA graph support
```

## Performance Considerations

1. **Guard Overhead**: Complex guards can slow down compilation checks
2. **Recompilation**: Changes to input shapes/dtypes may trigger recompilation
3. **Memory**: Inductor's buffer reuse optimization reduces memory allocation
4. **Kernel Launch**: Triton kernels are JIT compiled, causing first-run overhead
5. **Fallback**: Graph breaks cause fallback to eager mode, reducing performance

## Common Issues

1. **Graph Breaks**: Dynamo cannot capture certain Python constructs
   - Solution: Refactor code to avoid graph breaks

2. **Compilation Failures**: Complex operations may fail during lowering
   - Solution: Use `torch.compile(..., fullgraph=False)` for partial compilation

3. **Memory Usage**: Aggressive fusion may increase peak memory
   - Solution: Tune `config.max_fusion_size` and `config.memory_budget`

## Additional Architecture Diagrams

### Dispatch System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PyTorch Dispatcher                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Operator Call: at::sin(x)                                           │   │
│  │                                                                     │   │
│  │ 1. Dispatcher Lookup                                                │   │
│  │    ┌──────────────────────────────────────────────────────────┐    │   │
│  │    │ • Operator Registry (native_functions.yaml)              │    │   │
│  │    │ • Dispatch Key Extraction                                │    │   │
│  │    │   - Device (CPU, CUDA)                                   │    │   │
│  │    │   - Layout (Strided, Sparse)                             │    │   │
│  │    │   - Dtype (Float, Int)                                   │    │   │
│  │    └──────────────────────────────────────────────────────────┘    │   │
│  │                                                                     │   │
│  │ 2. Kernel Selection                                                 │   │
│  │    ┌──────────────────────────────────────────────────────────┐    │   │
│  │    │ Backend Priority:                                          │    │   │
│  │    │ 1. Backend-specific (CUDA, CPU)                          │    │   │
│  │    │ 2. CompositeExplicitAutograd                             │    │   │
│  │    │ 3. CompositeImplicitAutograd                             │    │   │
│  │    │ 4. Fallback                                                │    │   │
│  │    └──────────────────────────────────────────────────────────┘    │   │
│  │                                                                     │   │
│  │ 3. Kernel Execution                                                 │   │
│  │    ┌────────────────────┐          ┌────────────────────┐          │   │
│  │    │  sin_cuda(Tensor)  │    OR    │  sin_cpu(Tensor)   │          │   │
│  │    └────────────────────┘          └────────────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Note: Compiled kernels from Inductor register themselves in the dispatcher │
│        with higher priority for captured graphs                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      Dispatch Flow with torch.compile                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ User Code: torch.sin(x)                                             │   │
│  │                                                                     │   │
│  │ 1. Python API Layer  ──────────────────────┐                        │   │
│  │    torch/csrc/autograd/python_variable.cpp │                        │   │
│  │                                             │                        │   │
│  │ 2. Dispatcher Entry  ◄──────────────────────┘                        │   │
│  │    at::sin(Tensor) → Dispatcher::call()                              │   │
│  │                                                                     │   │
│  │ 3. Dispatch Key Resolution                                           │   │
│  │    auto dispatchKey = backendFallbackDispatchKey() + device + layout │   │
│  │                                                                     │   │
│  │ 4. Kernel Lookup                                                     │   │
│  │    KernelFunction* kernel = operatorLookupTable[dispatchKey]        │   │
│  │                                                                     │   │
│  │ 5. Two Paths:                                                       │   │
│  │    ┌────────────────────┐          ┌────────────────────┐          │   │
│  │    │  Eager Mode        │          │  Compiled Mode     │          │   │
│  │    │  (if not compiled) │          │  (if compiled)     │          │   │
│  │    └────────┬───────────┘          └────────┬───────────┘          │   │
│  │              │                              │                        │   │
│  │              ▼                              ▼                        │   │
│  │    ┌──────────────────┐          ┌──────────────────┐                │   │
│  │    │ Native Kernel    │          │ Compiled Kernel  │                │   │
│  │    │ sin_dispatch_... │          │ triton_sin_...   │                │   │
│  │    └──────────────────┘          └──────────────────┘                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory Management

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PyTorch Memory Architecture                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Tensor (Python)                                                    │   │
│  │    │                                                               │   │
│  │    └─▶  torch::Tensor (C++)  ────────────────────────────────────┤   │
│  │            │ ── storage_ (intrusive_ptr)                           │   │
│  │            │ ── sizes/strides                                     │   │
│  │            └─▶  autograd_meta_ (grad, grad_fn)                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
┌─────────────────────────────────────────────────────────────────────────────┐
│  Storage System                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ c10::TensorImpl                                                    │   │
│  │    │                                                               │   │
│  │    │ ── storage_ (c10::Storage)                                   │   │
│  │    │ ── sizes/strides/offset                                       │   │
│  │    │ ── data_type_                                                 │   │
│  │    └─▶  autograd_meta_                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
┌─────────────────────────────────────────────────────────────────────────────┐
│  Storage and Allocator                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ c10::Storage                                                       │   │
│  │    │ ── data_ptr_                                                  │   │
│  │    └─▶  storage_impl_ (intrusive_ptr)                            │   │
│  │                                                                    │   │
│  │ c10::StorageImpl                                                   │   │
│  │    │ ── data_ptr_ (void*)                                         │   │
│  │    │ ── size_bytes_                                               │   │
│  │    │ ── allocator_ (DeviceAllocator)                              │   │
│  │    │ ── device_ (Device)                                          │   │
│  │    └─▶  data_ptr_ may be nullptr (unallocated)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
┌─────────────────────────────────────────────────────────────────────────────┐
│  Allocator Hierarchy                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ c10::Allocator (Abstract Base)                                     │   │
│  │    │                                                               │   │
│  │    ├─▶  CPUAllocator (malloc)                                     │   │
│  │    ├─▶  CUDAAllocator (cudaMalloc)                                │   │
│  │    │    │                                                         │   │
│  │    │    ├─▶  NativeCUDAAllocator (direct cudaMalloc)            │   │
│  │    │    └─▶  CachedCUDAAllocator (caching + pooling)            │   │
│  │    │         │                                                    │   │
│  │    │         └─▶  CUDAPluggableAllocator (custom allocators)    │   │
│  │    │                                                              │   │
│  │    └─▶  MetaAllocator (meta tensors)                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      Memory Layout for 2D Tensor                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Tensor: torch.randn(3, 4)                                          │   │
│  │                                                                     │   │
│  │ sizes = [3, 4]                                                      │   │
│  │ strides = [4, 1]  (contiguous)                                      │   │
│  │ storage_offset = 0                                                  │   │
│  │                                                                     │   │
│  │ Memory Layout:                                                      │   │
│  │ ┌───────────────┐                                                   │   │
│  │ │ t[0, 0]       │ ◄── storage.data_ptr()                           │   │
│  │ ├───────────────┤                                                   │   │
│  │ │ t[0, 1]       │ ◄── stride[1] = 1 element                         │   │
│  │ ├───────────────┤                                                   │   │
│  │ │ t[0, 2]       │                                                   │   │
│  │ ├───────────────┤                                                   │   │
│  │ │ t[0, 3]       │                                                   │   │
│  │ ├───────────────┤                                                   │   │
│  │ │ t[1, 0]       │ ◄── stride[0] = 4 elements                        │   │
│  │ ├───────────────┤                                                   │   │
│  │ │ ...           │                                                   │   │
│  │ └───────────────┘                                                   │   │
│  │                                                                     │   │
│  │ Key Properties:                                                     │   │
│  │ - Storage is always contiguous (may have offset)                    │   │
│  │ - Tensor can be non-contiguous view                                  │   │
│  │ - Allocation size = 3 * 4 * 4 bytes (float32)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Autograd System Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Autograd Forward Pass                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ User Code: y = torch.sin(x)                                         │   │
│  │                                                                     │   │
│  │ 1. Wrapper Creation                                                 │   │
│  │    sin(x: Tensor)                                                  │   │
│  │    └─▶  at::sin(at::Tensor)  (C++)                               │   │
│  │         │                                                          │   │
│  │         └─▶  Applies to Tensor with requires_grad=True           │   │
│  │                                                                     │   │
│  │ 2. Node Creation                                                    │   │
│  │    ┌────────────────────┐                                         │   │
│  │    │  sin_backward      │ ◄── From derivatives.yaml              │   │
│  │    │  function          │    └── sin: self.conj().mul(grad.conj())│   │
│  │    └────────┬───────────┘                                         │   │
│  │              │                                                     │   │
│  │              ▼                                                     │   │
│  │    ┌──────────────────┐                                           │   │
│  │    │ autograd::Node   │  ← New node in computation graph        │   │
│  │    │   (SinBackward)  │                                           │   │
│  │    └────────┬─────────┘                                           │   │
│  │              │                                                     │   │
│  │              ▼                                                     │   │
│  │    ┌──────────────────┐                                           │   │
│  │    │  ctx (context)   │  ← Saves input for backward              │   │
│  │    │  - saved_tensors │     y.backward()                         │   │
│  │    │  - needs_input_grad│                                           │   │
│  │    └──────────────────┘                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Autograd Backward Pass                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ User Code: y.backward()                                             │   │
│  │                                                                     │   │
│  │ 1. Engine Start                                                     │   │
│  │    torch::autograd::Engine::execute(...)                           │   │
│  │    └─▶  ReadyQueue initialization                                  │   │
│  │                                                                     │   │
│  │ 2. Topological Sort                                                 │   │
│  │    ┌────────────────────┐                                         │   │
│  │    │ Computation Graph  │                                         │   │
│  │    │                    │   y (loss)                              │   │
│  │    │   SinBackward ◄────┼─────────┐                               │   │
│  │    │        ▲           │         │ backward()                    │   │
│  │    │        │           │         ▼                               │   │
│  │    │   x ◄──┼───────────┴─────────┘                               │   │
│  │    └────────┼──────────────────────── Topological order           │   │
│  │              │                                                     │   │
│  │ 3. Backward Pass Execution                                          │   │
│  │    foreach node in reversed(topological_order):                    │   │
│  │        node.apply(grad_output)                                     │   │
│  │        → calls SinBackward::apply()                                │   │
│  │                                                                     │   │
│  │ 4. Gradient Accumulation                                            │   │
│  │    x.grad += grad_input  ← Accumulate if multiple paths           │   │
│  │                                                                     │   │
│  │ 5. Cleanup                                                          │   │
│  │    Graph is freed (unless retain_graph=True)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      autograd.Function Implementation                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ class MyFunction(autograd.Function):                               │   │
│  │     @staticmethod                                                   │   │
│  │     def forward(ctx, x, y):                                        │   │
│  │         # 1. Save for backward                                     │   │
│  │         ctx.save_for_backward(x, y)                               │   │
│  │                                                                     │   │
│  │         # 2. Compute output                                        │   │
│  │         result = x * y                                             │   │
│  │         return result                                              │   │
│  │                                                                     │   │
│  │     @staticmethod                                                   │   │
│  │     def backward(ctx, grad_output):                                │   │
│  │         # 1. Retrieve saved tensors                                │   │
│  │         x, y = ctx.saved_tensors                                   │   │
│  │                                                                     │   │
│  │         # 2. Compute gradients                                     │   │
│  │         grad_x = grad_output * y                                   │   │
│  │         grad_y = grad_output * x                                   │   │
│  │         return grad_x, grad_y                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Usage:                                                                      │
│      z = MyFunction.apply(x, y)  ← Creates autograd graph                 │
│      z.backward()                ← Triggers backward pass                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Debugging Guide

### Common Issues and How to Debug Them

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Graph Breaks in Dynamo                                                  │
│                                                                             │
│ Symptom: "Graph break: call_function UserDefinedObjectVariable"           │
│                                                                             │
│ Debugging Steps:                                                            │
│ ═══════════════════════════════════════════════════════════════════════     │
│                                                                             │
│ 1. Enable verbose logging:                                                  │
│    import torch._dynamo.config                                              │
│    torch._dynamo.config.verbose = True                                      │
│                                                                             │
│ 2. Identify the breaking operation:                                        │
│    The error message will show which Python operation caused the break     │
│                                                                             │
│ 3. Common fixes:                                                            │
│    - Move custom Python classes outside compiled region                    │
│    - Use torch-compatible operations                                       │
│    - Refactor control flow                                                 │
│                                                                             │
│ 4. Suppress graph breaks for experimentation:                              │
│    torch.compile(..., fullgraph=True)  # Will error on graph break         │
│    torch.compile(..., fullgraph=False) # Allows graph breaks               │
│                                                                             │
│ File locations for debugging:                                              │
│ - torch/_dynamo/exc.py: Graph breaking exceptions                          │
│ - torch/_dynamo/symbolic_convert.py: Graph break detection                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Compilation Errors in Inductor                                          │
│                                                                             │
│ Symptom: "CompilationError: Failed to compile FX graph"                   │
│                                                                             │
│ Debugging Steps:                                                            │
│ ═══════════════════════════════════════════════════════════════════════     │
│                                                                             │
│ 1. Enable debug output:                                                    │
│    import torch._inductor.config                                           │
│    torch._inductor.config.debug = True                                     │
│                                                                             │
│ 2. Dump intermediate representations:                                      │
│    TORCHINDUCTOR_DUMP_FX_GRAPH=1  # Dump FX graph                         │
│    TORCHINDUCTOR_DUMP_SCHEDULER=1 # Dump scheduler IR                     │
│    TORCHINDUCTOR_DUMP_CODE=1      # Dump generated code                   │
│                                                                             │
│ 3. Check Triton compilation:                                              │
│    The error usually occurs in Triton code generation                     │
│    Look for errors in generated .py files                                 │
│                                                                             │
│ 4. Disable problematic passes:                                           │
│    torch._inductor.config.max_fusion_size = 0  # Disable fusion          │
│    torch._inductor.config.triton.cudagraphs = False                      │
│                                                                             │
│ File locations:                                                            │
│ - torch/_inductor/debug.py: Debug utilities                               │
│ - torch/_inductor/codegen/triton.py: Triton code gen                     │
│ - /tmp/torchinductor_$USER/: Generated code cache                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Memory Issues                                                           │
│                                                                             │
│ Symptom: OOM during compilation or runtime                                │
│                                                                             │
│ Debugging Steps:                                                            │
│ ═══════════════════════════════════════════════════════════════════════     │
│                                                                             │
│ 1. Check Inductor memory planning:                                        │
│    torch._inductor.config.memory_budget = 4 * 1024**3  # 4GB limit       │
│                                                                             │
│ 2. Reduce fusion:                                                        │
│    torch._inductor.config.max_fusion_size = 5  # Default is 64           │
│                                                                             │
│ 3. Enable memory profiling:                                              │
│    import torch._dynamo.utils                                             │
│    with torch._dynamo.utils.counters["inductor"]:                         │
│        # Run compiled function                                            │
│        pass                                                               │
│                                                                             │
│ 4. Monitor memory per operation:                                         │
│    torch.cuda.memory_summary()                                           │
│                                                                             │
│ Key files:                                                                 │
│ - torch/_inductor/memory.py: Memory planning                            │
│ - torch/_inductor/scheduler.py: Reuse analysis                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Performance Issues                                                      │
│                                                                             │
│ Symptom: Compiled code is slower than eager mode                          │
│                                                                             │
│ Debugging Steps:                                                            │
│ ═══════════════════════════════════════════════════════════════════════     │
│                                                                             │
│ 1. Profile with PyTorch profiler:                                        │
│    with torch.profiler.profile() as prof:                                │
│        compiled_fn(*args)                                                 │
│    prof.export_chrome_trace("trace.json")                                │
│                                                                             │
│ 2. Check for excessive recompilation:                                    │
│    import torch._dynamo.utils                                            │
│    print(torch._dynamo.utils.counters["frames"]["total"])               │
│    print(torch._dynamo.utils.counters["frames"]["ok"])                  │
│                                                                             │
│ 3. Verify kernel execution:                                              │
│    TORCHINDUCTOR_PROFILE=1  # Print kernel timing                        │
│                                                                             │
│ 4. Disable cudagraphs if causing issues:                                 │
│    torch._inductor.config.triton.cudagraphs = False                      │
│                                                                             │
│ 5. Check guard evaluation overhead:                                      │
│    TORCHDYNAMO_REPRO_AFTER="dynamo"  # Profile guard checks             │
│                                                                             │
│ Important metrics:                                                       │
│ - Compilation time vs. runtime                                           │
│ - Kernel launch overhead                                                 │
│ - Memory bandwidth utilization                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Testing Guide

### How to Write Tests for Compiled Code

```python
import torch
import torch._dynamo
from torch.testing._internal.common_utils import TestCase, run_tests

class TestCompiledOperators(TestCase):
    def test_basic_compile(self):
        # Test basic compilation
        @torch.compile
        def fn(x, y):
            return torch.sin(x) + y

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        # Check correctness
        result = fn(x, y)
        expected = torch.sin(x) + y
        self.assertEqual(result, expected)

    def test_compile_with_backward(self):
        # Test compilation with gradient computation
        @torch.compile
        def fn(x):
            return torch.sin(x).sum()

        x = torch.randn(4, 4, requires_grad=True)

        # Forward pass
        loss = fn(x)

        # Backward pass
        loss.backward()

        # Check gradient exists
        self.assertIsNotNone(x.grad)

    def test_dynamic_shapes(self):
        # Test with dynamic shapes
        @torch.compile(dynamic=True)
        def fn(x):
            return x * 2

        # Different shapes should reuse compiled kernel
        x1 = torch.randn(2, 3)
        x2 = torch.randn(4, 5)  # Should not trigger recompilation

        r1 = fn(x1)
        r2 = fn(x2)

        self.assertEqual(r1, x1 * 2)
        self.assertEqual(r2, x2 * 2)

    def test_graph_break_handling(self):
        # Test handling of graph breaks
        def fn(x):
            print("This causes graph break")  # graph break
            return torch.sin(x)

        compiled_fn = torch.compile(fn, fullgraph=False)
        x = torch.randn(4)

        # Should not error, but may have multiple graphs
        result = compiled_fn(x)
        self.assertEqual(result, torch.sin(x))

if __name__ == "__main__":
    run_tests()
```

### Testing Best Practices

1. **Always test both eager and compiled modes**:
   ```python
   def test_operator(self):
       def fn(x):
           return torch.my_op(x)

       x = torch.randn(4, 4)

       # Test eager
       eager_result = fn(x)

       # Test compiled
       compiled_fn = torch.compile(fn)
       compiled_result = compiled_fn(x)

       # Compare
       self.assertEqual(eager_result, compiled_result)
   ```

2. **Test different input types**:
   - CPU and CUDA tensors
   - Different dtypes (float32, float64, bfloat16)
   - Different layouts (contiguous, non-contiguous)

3. **Test gradient computation**:
   - Forward pass correctness
   - Backward pass correctness
   - Gradient accumulation
   - Higher-order gradients

4. **Test edge cases**:
   - Empty tensors
   - Scalar tensors
   - Large tensors (OOM handling)
   - In-place operations

5. **Use parameterized tests** for comprehensive coverage:
   ```python
   from torch.testing._internal.common_utils import parametrize

   @parametrize("dtype", [torch.float32, torch.float64])
   @parametrize("device", ["cpu", "cuda"])
   def test_parametrized(self, dtype, device):
       # Test with different combinations
       pass
   ```

## References

- [PyTorch 2.0: Faster Machine Learning Through Dynamic Python Bytecode Transformation](https://pytorch.org/get-started/pytorch-2.0/)
- [TorchDynamo Documentation](https://pytorch.org/docs/master/dynamo/)
- [Inductor Documentation](https://pytorch.org/docs/master/inductor/)
- [PyTorch GitHub Repository](https://github.com/pytorch/pytorch)
- [Debugging PyTorch 2.0 Compilation](https://pytorch.org/docs/master/torch.compiler.html#debugging)
