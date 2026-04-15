# PyTorch Compilation Stack Quick Reference

## Compilation Flow

```
Python Code → Dynamo → FX Graph → AOT → Inductor → Compiled Kernel
```

## Key Commands

### Debug Dynamo
```bash
# Enable verbose logging
TORCHDYNAMO_VERBOSE=1 python script.py

# Dump graphs
TORCHDYNAMO_REPRO_AFTER="dynamo" python script.py

# Suppress errors and continue
TORCHDYNAMO_SUPPRESS_ERRORS=1 python script.py
```

### Debug Inductor
```bash
# Dump intermediate representations
TORCHINDUCTOR_DUMP_FX_GRAPH=1
TORCHINDUCTOR_DUMP_SCHEDULER=1
TORCHINDUCTOR_DUMP_CODE=1
TORCHINDUCTOR_DUMP_POST_FUSION=1

# Profile kernels
TORCHINDUCTOR_PROFILE=1

# Debug specific pass
TORCHINDUCTOR_DUMP_AFTER_FUSION=1
```

### Environment Variables
```bash
# Cache directory
TORCHINDUCTOR_CACHE_DIR=/path/to/cache

# Force recompilation
TORCHDYNAMO_FORCE_RECOMPILE=1

# Disable compilation (fallback to eager)
TORCHDYNAMO_DISABLE=1
```

## Code Locations

### Core Implementation
- **Dynamo**: `torch/_dynamo/`
- **AOT**: `torch/_functorch/`
- **Inductor**: `torch/_inductor/`

### Key Files
- Graph capture: `torch/_dynamo/convert_frame.py`
- Guards: `torch/_dynamo/guards.py`
- AOT entry: `torch/_functorch/aot_autograd.py`
- Inductor lowering: `torch/_inductor/graph.py`
- Triton codegen: `torch/_inductor/codegen/triton.py`
- CPU codegen: `torch/_inductor/codegen/cpp.py`

## Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Graph break | Dynamo can't capture code | Refactor or use `fullgraph=False` |
| Compilation error | Inductor fails to compile | Enable debug dumps, check Triton code |
| OOM | Out of memory during compile | Reduce `max_fusion_size`, set `memory_budget` |
| Slow performance | Compiled slower than eager | Profile, check guards, reduce recompiles |
| Shape mismatch | Tensor shapes don't match | Check dynamic shapes, guards |

## Testing Commands

```bash
# Run specific test
python test/test_torch.py TestTorch.test_compile_add

# Run all dynamo tests
python test/test_dynamo.py

# Run inductor tests
python test/test_inductor.py

# With verbose output
pytest test/test_dynamo.py -v -k "test_basic"
```

## Configuration Snippets

### Basic Configuration
```python
import torch._dynamo.config
import torch._inductor.config

# Development settings
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False

# Inductor optimization
torch._inductor.config.debug = True
torch._inductor.config.max_fusion_size = 64

# Disable for debugging
torch._inductor.config.triton.cudagraphs = False
```

### Dynamic Shapes
```python
# Enable dynamic shape support
@torch.compile(dynamic=True)
def fn(x):
    return x * 2
```

### Debug Specific Part
```python
# Debug only AOT
with torch._dynamo.config.patch(only_aot=True):
    compiled_fn = torch.compile(fn)

# Debug only Inductor
with torch._inductor.config.patch(debug=True):
    compiled_fn = torch.compile(fn)
```

## Performance Tuning

```python
# Reduce compilation overhead
import torch._dynamo

torch._dynamo.config.cache_size_limit = 128  # More cached functions

# Reduce memory usage
torch._inductor.config.memory_budget = 2 * 1024**3  # 2GB limit

# Optimize for inference (no grad)
torch.set_grad_enabled(False)
model = torch.compile(model)

# Optimize for training
model = torch.compile(model, mode="reduce-overhead")
```

## Graph Break Patterns

### Causes Graph Breaks
```python
# ❌ Print statements
def fn(x):
    print(x)  # Graph break
    return x * 2

# ❌ Custom classes
class MyClass:
    def __call__(self, x):
        return x * 2

obj = MyClass()
def fn(x):
    return obj(x)  # Graph break

# ❌ Data dependent control flow
def fn(x):
    if x.sum() > 0:  # Graph break
        return x
    return -x
```

### Works with Compile
```python
# ✅ Tensor operations
@torch.compile
def fn(x):
    return torch.sin(x) + torch.cos(x)

# ✅ torch.where for data-dependent
@torch.compile
def fn(x):
    return torch.where(x.sum() > 0, x, -x)

# ✅ Static control flow
@torch.compile
def fn(x, flag):
    if flag:  # Static flag, not graph break
        return x
    return -x
```

## Useful Debug Functions

```python
import torch._dynamo
import torch._inductor

# Print compiled graphs
def print_graph(fn, *args):
    def debug_backend(gm, example_inputs):
        print("FX Graph:")
        gm.graph.print_tabular()
        return gm

    compiled = torch.compile(fn, backend=debug_backend)
    compiled(*args)

# Count recompiles
def count_recompiles(fn, *args_list):
    before = torch._dynamo.utils.counters["frames"].get("total", 0)
    for args in args_list:
        fn(*args)
    after = torch._dynamo.utils.counters["frames"].get("total", 0)
    print(f"Recompiles: {after - before}")

# Dump all intermediate representations
def dump_compilation(fn, *args):
    import os
    os.environ["TORCHINDUCTOR_DUMP_FX_GRAPH"] = "1"
    os.environ["TORCHINDUCTOR_DUMP_SCHEDULER"] = "1"
    os.environ["TORCHINDUCTOR_DUMP_CODE"] = "1"

    compiled = torch.compile(fn)
    compiled(*args)
```

## Documentation Links

- [PyTorch 2.0 Overview](https://pytorch.org/get-started/pytorch-2.0/)
- [torch.compile API](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [TorchDynamo Deep Dive](https://pytorch.org/docs/master/torch.compiler.html)
- [Inductor Triton Backend](https://pytorch.org/docs/master/torch.compiler.html#inductor-backend)
- [Debugging Guide](https://pytorch.org/docs/master/torch.compiler.html#debugging)
