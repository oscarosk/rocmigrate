"""
End-to-end integration tests for the Rocmigrate agent.

Tests the full porting loop on three kernels of increasing complexity:
saxpy (1D, simple), vector_dot (1D, reduction), matmul (2D, multi-array).

Usage:
    python tests/test_loop.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agent.main import port_cuda_to_hip
from mcp_servers.hipcc.server import compile_hip


# CUDA-only symbols that should never appear in the agent's output
CUDA_LEFTOVERS = [
    "cudaMalloc", "cudaFree", "cudaMemcpy",
    "cudaMemcpyHostToDevice", "cudaMemcpyDeviceToHost",
    "cuda_runtime.h", "cudaDeviceSynchronize",
]


def _check_port(kernel_name: str) -> str:
    """Run the agent on a kernel and assert the output is valid HIP."""
    cuda_path = ROOT / "kernels" / "cuda" / f"{kernel_name}.cu"
    cuda_source = cuda_path.read_text()

    print(f"[TEST] Porting {kernel_name}.cu...")
    hip_source = port_cuda_to_hip(cuda_source)

    # No markdown fences leaked through
    assert not hip_source.startswith("```"), (
        f"{kernel_name}: Markdown fence not stripped"
    )

    # Has the HIP runtime include
    assert "hip/hip_runtime.h" in hip_source, (
        f"{kernel_name}: Missing HIP runtime include"
    )

    # No CUDA leftovers
    for symbol in CUDA_LEFTOVERS:
        assert symbol not in hip_source, (
            f"{kernel_name}: Unported {symbol} found in output"
        )

    # Kernel definition preserved
    assert "__global__" in hip_source, (
        f"{kernel_name}: Kernel definition missing"
    )

    # Compiles via mock hipcc
    result = compile_hip(hip_source, kernel_name)
    assert result.success, (
        f"{kernel_name}: Mock compile failed: {result.errors}"
    )

    print(f"[PASS] {kernel_name} — {len(hip_source)} chars, compiles cleanly")
    return hip_source


def test_saxpy():
    """Simple 1D kernel — the smoke test."""
    _check_port("saxpy")


def test_vector_dot():
    """1D kernel with shared memory reduction — exercises atomicAdd."""
    _check_port("vector_dot")


def test_matmul():
    """2D kernel with multiple buffers — the realistic case."""
    _check_port("matmul")


if __name__ == "__main__":
    print("=" * 60)
    print("ROCMIGRATE AGENT — INTEGRATION TESTS (3 kernels)")
    print("=" * 60)
    print()

    try:
        test_saxpy()
        print()
        test_vector_dot()
        print()
        test_matmul()
        print()
        print("=" * 60)
        print("ALL TESTS PASSED — agent works on all 3 kernels")
        print("=" * 60)
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        sys.exit(1)