"""
Mock hipcc compiler MCP server.

Simulates the AMD HIP compiler (hipcc) for local development before
GPU credits land. Returns realistic compile results based on simple
heuristics — checks for valid HIP includes, common API patterns, and
syntactic red flags.

Once the MI300X droplet is live, this module is replaced by a real
implementation that SSHes into the droplet and invokes actual hipcc.
The agent code that calls this is unchanged.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class CompileResult:
    """Mirrors what a real hipcc invocation returns."""
    success: bool
    binary_path: str | None
    stdout: str
    stderr: str
    errors: List[str]
    warnings: List[str]


def compile_hip(source_code: str, output_name: str = "kernel") -> CompileResult:
    """
    Mock-compile HIP source code.

    In production this will execute:
        hipcc -o {output_name} {source_file}
    on the MI300X droplet via SSH. For now, we use simple heuristics
    to simulate realistic compile success/failure responses so the
    agent loop can be developed end-to-end.

    Args:
        source_code: HIP/ROCm C++ source as a string.
        output_name: Name for the (mock) output binary.

    Returns:
        CompileResult with success status and any errors/warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Heuristic 1: Must include the HIP runtime header
    if "hip/hip_runtime.h" not in source_code and "hip_runtime.h" not in source_code:
        errors.append(
            "fatal error: 'hip/hip_runtime.h' file not found. "
            "Did you forget to replace cuda_runtime.h?"
        )

    # Heuristic 2: Common CUDA-only symbols that indicate incomplete porting
    cuda_only_symbols = [
        "cudaMalloc",
        "cudaFree",
        "cudaMemcpy",
        "cudaMemcpyHostToDevice",
        "cudaMemcpyDeviceToHost",
        "cuda_runtime.h",
        "cudaDeviceSynchronize",
        "cudaGetLastError",
    ]
    for symbol in cuda_only_symbols:
        if symbol in source_code:
            errors.append(
                f"error: use of undeclared identifier '{symbol}'. "
                f"Did you mean '{symbol.replace('cuda', 'hip')}'?"
            )

    # Heuristic 3: Warn (but don't fail) on style issues
    if "__global__" in source_code and "hipLaunchKernelGGL" not in source_code and "<<<" not in source_code:
        warnings.append(
            "warning: kernel defined but no launch syntax detected"
        )

    # Build the result
    if errors:
        return CompileResult(
            success=False,
            binary_path=None,
            stdout="",
            stderr="\n".join(errors + warnings),
            errors=errors,
            warnings=warnings,
        )

    return CompileResult(
        success=True,
        binary_path=f"./{output_name}",
        stdout=f"hipcc: compiled {output_name} successfully (MOCK)",
        stderr="\n".join(warnings) if warnings else "",
        errors=[],
        warnings=warnings,
    )


# Quick self-test when run directly
if __name__ == "__main__":
    print("=" * 60)
    print("MOCK HIPCC SELF-TEST")
    print("=" * 60)

    # Test 1: Valid HIP code (the saxpy reference) should compile
    print("\nTest 1: Valid HIP code")
    print("-" * 60)
    with open("kernels/hip_reference/saxpy.hip", "r") as f:
        valid_hip = f.read()
    result = compile_hip(valid_hip, "saxpy")
    print(f"  success: {result.success}")
    print(f"  stdout:  {result.stdout}")
    if not result.success:
        print(f"  errors:  {result.errors}")

    # Test 2: CUDA code (unported) should fail with helpful errors
    print("\nTest 2: Unported CUDA code (should fail)")
    print("-" * 60)
    with open("kernels/cuda/saxpy.cu", "r") as f:
        cuda_code = f.read()
    result = compile_hip(cuda_code, "saxpy")
    print(f"  success: {result.success}")
    print(f"  errors:")
    for err in result.errors[:3]:  # first 3 errors only
        print(f"    - {err}")
    if len(result.errors) > 3:
        print(f"    ... and {len(result.errors) - 3} more")

    print("\n" + "=" * 60)
    print("Self-test complete.")
    print("=" * 60)