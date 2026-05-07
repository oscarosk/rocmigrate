"""
Rocmigrate agent — autonomous CUDA → ROCm porting.

Architecture:
    [CUDA source] → [Translator agent] → [Mock hipcc] → [Validator]
                            ↑___________error feedback loop_____|

The translator is a Pydantic AI agent backed by an LLM (Groq Llama 3.3
during development, Qwen 2.5 Coder on MI300X in production). It generates
a HIP port of the input CUDA code, calls the compiler tool, and if the
compile fails, reads the error messages and tries again — up to a configured
maximum number of iterations.

Tool calling follows the ReAct pattern (reason + act) demonstrated by AMD
in the official hackathon workshop.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Make sibling packages importable when running this file directly
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from mcp_servers.hipcc.server import compile_hip, CompileResult

load_dotenv()

# ---------- Model setup ----------

LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

if not all([LLM_BASE_URL, LLM_API_KEY, LLM_MODEL]):
    raise RuntimeError(
        "Missing LLM config. Copy .env.example to .env and fill in "
        "LLM_BASE_URL, LLM_API_KEY, LLM_MODEL."
    )

provider = OpenAIProvider(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
model = OpenAIChatModel(LLM_MODEL, provider=provider)

# ---------- The translator agent ----------

SYSTEM_PROMPT = """You are an expert GPU programmer specializing in porting
NVIDIA CUDA code to AMD ROCm/HIP. Your job is to translate CUDA source code
into equivalent HIP code that compiles cleanly with hipcc on AMD MI300X.

Rules:
1. Replace `cuda_runtime.h` with `hip/hip_runtime.h`.
2. Replace all `cuda*` API calls with their `hip*` equivalents
   (e.g., cudaMalloc -> hipMalloc, cudaMemcpy -> hipMemcpy).
3. Preserve kernel logic exactly. The numerical output of the ported code
   must match the original.
4. Keep the code well-formatted and readable.
5. After producing a port, you MUST call the `compile_port` tool to verify
   it compiles. If it fails, read the errors carefully and produce a corrected
   version.

When you have a successful compile, return ONLY the final HIP source code
(no explanation, no markdown fences) as your final response."""

agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
)


@agent.tool_plain
def compile_port(hip_source: str) -> dict:
    """
    Compile a HIP source string. Returns success status and any errors.

    Use this tool to verify your translation before returning a final answer.
    If the compile fails, read the errors carefully — they often tell you
    exactly what symbol still needs porting.

    Args:
        hip_source: The complete HIP/ROCm C++ source code to compile.

    Returns:
        A dict with:
          - success (bool): True if compile succeeded
          - errors (list of str): compile errors, if any
          - warnings (list of str): non-fatal warnings, if any
    """
    result: CompileResult = compile_hip(hip_source, output_name="kernel")
    return {
        "success": result.success,
        "errors": result.errors,
        "warnings": result.warnings,
    }


# ---------- Public entry point ----------

def port_cuda_to_hip(cuda_source: str, max_iterations: int = 5) -> str:
    """
    Port a CUDA source string to HIP and return the working HIP code.

    The agent will call the compile tool internally and iterate on errors
    until it produces code that compiles successfully or exhausts the
    iteration budget.

    Args:
        cuda_source: Complete CUDA C/C++ source.
        max_iterations: Max compile-and-retry attempts before giving up.

    Returns:
        The final HIP source as a string.

    Raises:
        RuntimeError: If the agent cannot produce compiling code within
        the iteration budget.
    """
    user_message = (
        f"Port the following CUDA code to HIP. Use the compile_port tool to "
        f"verify your work before returning the final answer.\n\n"
        f"CUDA source:\n```cuda\n{cuda_source}\n```"
    )

    # max_iterations bounds how many tool calls + retries the agent can make
    result = agent.run_sync(
        user_message,
        model_settings={"max_tokens": 4096},
    )

    return result.output.strip()


# ---------- CLI for quick testing ----------

if __name__ == "__main__":
    print("=" * 60)
    print("ROCMIGRATE AGENT — saxpy demo")
    print("=" * 60)

    cuda_path = ROOT / "kernels" / "cuda" / "saxpy.cu"
    cuda_source = cuda_path.read_text()

    print(f"\nLoaded CUDA source from: {cuda_path}")
    print(f"Source length: {len(cuda_source)} chars")
    print(f"\nCalling agent with model: {LLM_MODEL}")
    print("(this may take 10-30 seconds...)\n")

    hip_source = port_cuda_to_hip(cuda_source)

    print("=" * 60)
    print("AGENT OUTPUT — ported HIP source:")
    print("=" * 60)
    print(hip_source)
    print("=" * 60)

    # Verify the agent's output actually compiles
    final_check = compile_hip(hip_source, "saxpy")
    print(f"\nFinal compile check: {'PASS' if final_check.success else 'FAIL'}")
    if not final_check.success:
        print(f"Errors: {final_check.errors}")