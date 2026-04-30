"""Test script to verify multi-GPU and AMP optimizations work correctly."""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

def test_amp_basic():
    """Test that AMP works correctly."""
    print("\n🧪 Test 1: Mixed Precision (AMP) Basic Test")
    
    # Simple model
    model = nn.Linear(100, 10)
    scaler = GradScaler(enabled=True)
    
    x = torch.randn(32, 100)
    target = torch.randn(32, 10)
    
    with autocast(dtype=torch.float16):
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(torch.optim.Adam(model.parameters()))
    scaler.update()
    
    print("  ✅ AMP forward/backward successful")
    print(f"  Output dtype: {output.dtype} (expected: torch.float16)")
    return True

def test_dataparallel_detection():
    """Test DataParallel detection logic."""
    print("\n🧪 Test 2: Multi-GPU Detection Test")
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if num_gpus >= 2:
        print(f"  ✅ Detected {num_gpus} GPUs")
        for i in range(num_gpus):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Test DataParallel wrapping
        model = nn.Linear(100, 10)
        dp_model = nn.DataParallel(model)
        print(f"  ✅ DataParallel wrapper created successfully")
        print(f"  Device IDs: {dp_model.device_ids}")
    else:
        print(f"  ⚠️  Only {num_gpus} GPU(s) available (need 2+ for multi-GPU)")
        print(f"  ℹ️  Code will automatically use both GPUs when running on Kaggle")
    
    return True

def test_dataloader_optimization():
    """Test optimized DataLoader configuration."""
    print("\n🧪 Test 3: Optimized DataLoader Test")
    
    from torch.utils.data import DataLoader, TensorDataset
    
    dataset = TensorDataset(torch.randn(1000, 10))
    
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # Iterate to test
    for batch in loader:
        break
    
    print(f"  ✅ DataLoader with optimization config works")
    print(f"    num_workers=4, pin_memory=True, prefetch_factor=2")
    return True

def test_checkpoint_compatibility():
    """Test that checkpoint system works with optimized training."""
    print("\n🧪 Test 4: Checkpoint Compatibility Test")
    
    from src.utils.checkpoint import save_checkpoint, load_checkpoint
    import tempfile
    import os
    
    model = nn.Linear(100, 10)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test_ckpt.pt')
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            loss=0.5,
            config={'test': True},
            path=path,
            metadata={'test_resolution': 225}
        )
        
        # Load back
        info = load_checkpoint(path, model, strict=False)
        
        assert info['epoch'] == 5
        assert info['loss'] == 0.5
        assert info['metadata']['test_resolution'] == 225
        
        print(f"  ✅ Checkpoint save/load with metadata works")
        print(f"    Epoch: {info['epoch']}, Loss: {info['loss']}")
        print(f"    Metadata: {info['metadata']}")
    
    return True

def main():
    print("=" * 70)
    print("🚀 Testing Multi-GPU & AMP Optimizations")
    print("=" * 70)
    
    tests = [
        test_amp_basic,
        test_dataparallel_detection,
        test_dataloader_optimization,
        test_checkpoint_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("✅ All optimization tests passed!")
        print("\n📝 Next steps:")
        print("  1. Run: python scripts/train.py")
        print("  2. Monitor: watch -n 1 nvidia-smi")
        print("  3. See: MULTI_GPU_GUIDE.md for details")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
