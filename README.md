# Rocmigrate

> Autonomous AI agent that ports CUDA kernels to AMD ROCm, compiles and tests them on MI300X, and ships working code.

Built for the [AMD Developer Hackathon](https://lablab.ai/event/amd-developer-hackathon), May 2026.

## The problem

Migrating GPU code from NVIDIA's CUDA to AMD's ROCm is the single largest commercial barrier between AMD and NVIDIA's data-center revenue. AMD's official tool, HIPIFY, handles syntactic translation but fails on:

- Performance-sensitive patterns that need rethinking for CDNA architecture
- CUDA-specific intrinsics without direct ROCm equivalents
- Complex macros and template-heavy code
- Anything requiring semantic understanding of what the kernel *does*

## The solution

Rocmigrate is a multi-agent system that starts where HIPIFY gives up. The agent reasons about a CUDA kernel, generates a HIP port, compiles it on MI300X, runs it against the original output, and iterates until correct. A final optimizer stage tunes the working port for AMD's CDNA architecture (LDS usage, wavefront sizing, memory coalescing).

## Architecture

## Architecture
[CUDA kernel] → [Analyzer] → [Translator] → [Compiler (hipcc)] → [Runner (MI300X)] → [Validator] → [Optimizer] → [ROCm kernel + benchmark]
↑___________________________________|
(critic loop on failure)

Built with:

- **Pydantic AI** — agent orchestration (per AMD's recommended workshop pattern)
- **MCP** — Model Context Protocol for tool integration
- **vLLM + Qwen 2.5 Coder** — open-source code-specialized LLM, served on AMD Instinct MI300X
- **AMD ROCm 7.2** — compute stack
- **Gradio** — demo UI, deployed as a Hugging Face Space

## Status

🚧 In active development for the AMD Developer Hackathon. Submission deadline May 10, 2026.

## License

MIT