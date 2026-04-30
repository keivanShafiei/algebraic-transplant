# Multi-GPU & Mixed Precision Training Guide

## Summary of Optimizations for Dual Tesla T4 Setup

This document describes the optimizations applied to leverage your dual Tesla T4 GPU setup (15GB each, 31GB system RAM).

### Applied Optimizations

#### 1. **Multi-GPU Training with DataParallel**
- Automatically detects and uses both GPUs when available
- Splits batches across GPUs for parallel processing
- Effective batch size doubles without memory issues

**Expected Benefits:**
- ~1.8x speedup on training throughput
- Ability to use larger batch sizes (32+ instead of 16)

#### 2. **Mixed Precision Training (AMP)**
- Uses `torch.cuda.amp.autocast` for FP16 compute
- `GradScaler` prevents underflow during backpropagation
- Reduces VRAM usage by ~40-50%

**Expected Benefits:**
- 2-3x faster matrix operations on Tensor Cores
- Reduced memory footprint enables higher resolutions
- No loss in accuracy (automatic loss scaling)

#### 3. **Optimized DataLoader**
```python
DataLoader(
    num_workers=4,      # Parallel data loading
    pin_memory=True,    # Faster CPU→GPU transfer
    prefetch_factor=2   # Prefetch batches
)
```

**Expected Benefits:**
- Eliminates data loading bottlenecks
- Keeps GPUs saturated with work
- Better utilization of 31GB system RAM

### Configuration Recommendations

#### For Your Current Setup (2× T4, 15GB each):

**Option A: Higher Resolution Training**
```yaml
# config.yaml
n_nodes_list: [400, 1000, 2500, 5000]  # Increased from [225, 1000, 5000, 10000]
batch_size: 16                          # Same, but more headroom
```

**Option B: Larger Batch Size**
```yaml
# config.yaml
batch_size: 32  # Doubled for better multi-GPU utilization
epochs: 150     # Fewer epochs needed with larger batches
```

**Option C: Balanced Approach (Recommended)**
```yaml
# config.yaml
n_nodes_list: [400, 1000, 2500]
batch_size: 24
epochs: 180
```

### Memory Usage Estimates

| Configuration | Single GPU VRAM | Multi-GPU VRAM (each) | Notes |
|--------------|-----------------|----------------------|-------|
| N=225, BS=16, FP32 | ~2.5 GB | ~1.5 GB | Baseline |
| N=225, BS=16, FP16 | ~1.5 GB | ~0.9 GB | AMP only |
| N=400, BS=16, FP16 | ~3.2 GB | ~1.8 GB | Recommended |
| N=1000, BS=16, FP16 | ~7.5 GB | ~4.2 GB | High-res |
| N=2500, BS=8, FP16 | ~12 GB | ~7 GB | Max resolution |

### Running Training

**Standard Training (auto-detects GPUs):**
```bash
cd /workspace/repo
python scripts/train.py
```

**Expected Output:**
```
🚀 Using 2 GPUs with DataParallel
  GPU 0: Tesla T4
  GPU 1: Tesla T4
✅ Multi-GPU enabled (2 GPUs)
✅ Mixed Precision (AMP) enabled
```

### Monitoring GPU Usage

```bash
# Watch GPU utilization in real-time
watch -n 1 nvidia-smi

# Or use nvtop for interactive monitoring
nvtop
```

### Troubleshooting

#### Issue: CUDA Out of Memory
**Solution:** Reduce batch size or resolution:
```yaml
batch_size: 8  # Instead of 16
n_nodes_list: [225, 400]  # Lower resolutions
```

#### Issue: Low GPU Utilization (<50%)
**Solution:** Increase batch size or enable more workers:
```yaml
batch_size: 32
```
In `train.py`, increase `num_workers=8` if CPU has enough cores.

#### Issue: Data Loading Bottleneck
**Solution:** Ensure data is on fast storage and increase prefetch:
```python
DataLoader(..., prefetch_factor=4, num_workers=8)
```

### Performance Benchmarks (Expected)

| Metric | Single GPU (FP32) | Dual GPU (FP16) | Improvement |
|--------|-------------------|-----------------|-------------|
| Samples/sec (N=225) | ~45 | ~150 | 3.3x |
| Samples/sec (N=1000) | ~12 | ~45 | 3.7x |
| VRAM Usage | 2.5 GB | 1.5 GB | 40% reduction |
| Training Time (200 epochs, N=225) | ~2.5 hrs | ~45 min | 3.3x faster |

### Next Steps

1. **Test with current config:**
   ```bash
   python scripts/train.py
   ```

2. **Monitor GPU usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Adjust based on observations:**
   - If VRAM < 50% used → increase batch_size or resolution
   - If GPU util < 70% → increase batch_size
   - If training too fast → increase epochs or resolution

4. **For production runs:**
   Consider using `torch.distributed` instead of `DataParallel` for even better scaling.

### Advanced: Distributed Training (Future)

For maximum performance, consider migrating to `torch.distributed`:
```python
# Replace DataParallel with DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Launch with: torchrun --nproc_per_node=2 scripts/train.py
```

This can provide additional 10-20% speedup over DataParallel.
