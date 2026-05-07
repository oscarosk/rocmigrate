"""
End-to-end integration test for the Rocmigrate agent.

Verifies that the agent can take CUDA source, produce HIP source,
and that the HIP source passes mock compilation. Run this anytime
you want to confirm the loop still works.

Usage:
    python tests/test_loop.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agent.main import port_cuda_to_hip
from mcp_servers.hipcc.server import compile_hip


def test_saxpy_port():
    """Agent should successfully port saxpy.cu to compiling HIP."""
    cuda_path = ROOT / "kernels" / "cuda" / "saxpy.cu"
    cuda_source = cuda_path.read_text()

    print(f"[TEST] Porting {cuda_path.name}...")
    hip_source = port_cuda_to_hip(cuda_source)

    # Sanity checks on the output shape
    assert hip_source, "Agent returned empty output"
    assert "hip/hip_runtime.h" in hip_source, "Missing HIP runtime include"
    assert "cudaMalloc" not in hip_source, "Unported cudaMalloc found"
    assert "cudaMemcpy" not in hip_source, "Unported cudaMemcpy found"
    assert "cuda_runtime.h" not in hip_source, "Unported cuda_runtime.h found"
    assert "__global__" in hip_source, "Kernel definition missing"

    # The output must compile through our mock hipcc
    result = compile_hip(hip_source, "saxpy")
    assert result.success, f"Final compile failed: {result.errors}"

    print(f"[PASS] saxpy port — {len(hip_source)} chars, compiles cleanly")
    return hip_source


def test_agent_handles_clean_input():
    """Sanity check: agent shouldn't crash on already-ported HIP."""
    hip_path = ROOT / "kernels" / "hip_reference" / "saxpy.hip"
    hip_source_input = hip_path.read_text()

    print(f"[TEST] Feeding agent already-ported {hip_path.name}...")
    output = port_cuda_to_hip(hip_source_input)

    assert output, "Agent returned empty output on clean input"
    result = compile_hip(output, "saxpy_clean")
    assert result.success, f"Clean-input compile failed: {result.errors}"

    print(f"[PASS] Clean input handled — agent produced compiling output")


if __name__ == "__main__":
    print("=" * 60)
    print("ROCMIGRATE AGENT — INTEGRATION TESTS")
    print("=" * 60)

    try:
        test_saxpy_port()
        print()
        test_agent_handles_clean_input()
        print()
        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        sys.exit(1)