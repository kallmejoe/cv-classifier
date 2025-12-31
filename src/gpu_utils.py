"""GPU utilities for device detection and management.

This module provides utilities for detecting and managing GPU/CPU devices
for PyTorch-based training. It supports:
- NVIDIA CUDA GPUs
- Apple Metal (MPS) on macOS
- CPU fallback

Usage:
    from src.gpu_utils import get_device, get_device_info, is_gpu_available

    device = get_device()  # Automatically selects best available device
    print(get_device_info())  # Print device information
"""

import os
from typing import Optional, Dict, Any

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


def is_torch_available() -> bool:
    """Check if PyTorch is installed."""
    return TORCH_AVAILABLE


def get_device(prefer_gpu: bool = True, device_id: Optional[int] = None) -> "torch.device":
    """
    Get the best available device for PyTorch operations.

    Priority order:
    1. NVIDIA CUDA GPU (if available and prefer_gpu=True)
    2. Apple Metal MPS (if available and prefer_gpu=True, macOS only)
    3. CPU

    Args:
        prefer_gpu: If True, prefer GPU over CPU when available
        device_id: Specific CUDA device ID to use (only for multi-GPU systems)

    Returns:
        torch.device: The selected device

    Raises:
        ImportError: If PyTorch is not installed
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for GPU support. "
            "Install it with: pip install torch"
        )

    if not prefer_gpu:
        return torch.device("cpu")

    # Check for NVIDIA CUDA
    if torch.cuda.is_available():
        if device_id is not None:
            if device_id >= torch.cuda.device_count():
                print(f"Warning: CUDA device {device_id} not available, using device 0")
                device_id = 0
            return torch.device(f"cuda:{device_id}")
        return torch.device("cuda")

    # Check for Apple Metal (MPS) on macOS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Check if MPS is actually built and functional
        try:
            if torch.backends.mps.is_built():
                return torch.device("mps")
        except Exception:
            pass

    # Fallback to CPU
    return torch.device("cpu")


def is_gpu_available() -> bool:
    """
    Check if any GPU (CUDA or MPS) is available.

    Returns:
        bool: True if GPU is available, False otherwise
    """
    if not TORCH_AVAILABLE:
        return False

    # Check NVIDIA CUDA
    if torch.cuda.is_available():
        return True

    # Check Apple Metal (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            return torch.backends.mps.is_built()
        except Exception:
            pass

    return False


def get_cuda_device_count() -> int:
    """Get the number of available CUDA devices."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed information about available compute devices.

    Returns:
        Dict containing device information
    """
    info: Dict[str, Any] = {
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": False,
        "mps_available": False,
        "device_count": 0,
        "current_device": "cpu",
        "devices": []
    }

    if not TORCH_AVAILABLE:
        return info

    info["torch_version"] = torch.__version__

    # CUDA information
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["device_count"] = torch.cuda.device_count()

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count
            }
            info["devices"].append(device_info)

        info["current_device"] = f"cuda:{torch.cuda.current_device()}"

    # MPS information (Apple Metal)
    if hasattr(torch.backends, "mps"):
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                info["mps_available"] = True
                if not info["cuda_available"]:
                    info["current_device"] = "mps"
                    info["device_count"] = 1
                    info["devices"].append({
                        "id": 0,
                        "name": "Apple Metal GPU",
                        "type": "mps"
                    })
        except Exception:
            pass

    return info


def print_device_info() -> None:
    """Print formatted device information to console."""
    info = get_device_info()

    print("\n" + "="*60)
    print("COMPUTE DEVICE INFORMATION")
    print("="*60)

    if not info["torch_available"]:
        print("PyTorch is not installed.")
        print("Install with: pip install torch")
        return

    print(f"PyTorch version: {info.get('torch_version', 'unknown')}")

    if info["cuda_available"]:
        print(f"\n✓ NVIDIA CUDA is available")
        print(f"  CUDA version: {info.get('cuda_version', 'unknown')}")
        if info.get("cudnn_version"):
            print(f"  cuDNN version: {info['cudnn_version']}")
        print(f"  GPU count: {info['device_count']}")

        for device in info["devices"]:
            print(f"\n  Device {device['id']}: {device['name']}")
            print(f"    Memory: {device['total_memory_gb']:.1f} GB")
            print(f"    Compute capability: {device['compute_capability']}")
            print(f"    Multi-processors: {device['multi_processor_count']}")

    elif info["mps_available"]:
        print(f"\n✓ Apple Metal (MPS) is available")
        print("  Using Apple Silicon GPU")

    else:
        print("\n✗ No GPU available, using CPU")

    print(f"\nSelected device: {info['current_device']}")
    print("="*60 + "\n")


def optimize_cuda_settings() -> None:
    """
    Apply optimizations for CUDA training.

    This function sets various CUDA optimization flags for better performance.
    Should be called before training starts.
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return

    # Enable cuDNN auto-tuner for optimal convolution algorithms
    torch.backends.cudnn.benchmark = True

    # Use TensorFloat-32 for faster training on Ampere+ GPUs
    if hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True


def clear_gpu_memory() -> None:
    """
    Clear GPU memory cache.

    Useful for freeing up GPU memory between training runs.
    """
    if not TORCH_AVAILABLE:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_stats() -> Optional[Dict[str, float]]:
    """
    Get current GPU memory statistics.

    Returns:
        Dict with memory stats in GB, or None if CUDA not available
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None

    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "cached_gb": torch.cuda.memory_reserved() / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }


if __name__ == "__main__":
    print_device_info()
