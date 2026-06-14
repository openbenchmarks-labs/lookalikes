"""Lookalike benchmark runners + LLM-as-judge.

This package holds one runner per vendor (`runners/<vendor>.py`), a shared
LLM judge (`judge.py`), and the common dataclasses + snapshot I/O
(`common.py`) used by the orchestrator at
`scripts/run_lookalike_benchmark.py`.

Each runner exposes:

    def run(seed: Seed, k: int, config: dict) -> RunResult: ...
    CONFIGS: list[dict]   # named configs to sweep; orchestrator picks best

Each runner is responsible for hitting its vendor's lookalike endpoint
with the supplied config and returning at most `k` Candidate objects.
The orchestrator handles judging, best-config selection, and persistence.
"""
