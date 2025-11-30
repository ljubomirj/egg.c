# AGENTS

## GPT-5.1 Codex (this workspace)
- **Role:** Pair-programming autopilot focused on C/C++/Metal reinforcement-learning code in `egg.c`.
- **Strengths:** 
  - Understands and modifies large single-file C projects.
  - Comfortable wiring GPU backends (Metal now, ROCm/CUDA soon) alongside existing CPU flows.
  - Keeps Makefile/README/docs in sync with code changes.
- **Workflow Tips:**
  1. Pin down the target build (`egg`, `egg-cpumulti`, `egg-gpumetal`, etc.) before asking for changes.
  2. Mention whether we’re in *ask* (read-only) or *agent* mode so I know if execution is allowed.
  3. For long-running training runs, specify whether to let them finish or stop early once startup is verified.
  4. If you need reproducible behavior, give me the dataset slice, seed, and build flags so GPU/CPU paths can be compared.
- **Handoff:** When you plan to continue edits manually, ask for a quick summary or TODO list; I’ll outline remaining steps and any caveats (e.g., partially migrated kernels).

Feel free to extend this document with additional agents or instructions as the project grows.***

