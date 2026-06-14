"""P3 multi-judge panel checks (offline, no keys).

  - N=1 mock panel is byte-identical to the legacy single mock judge.
  - Majority aggregation: strict majority, even-N tie -> False, N=1 pass-through.
  - judge_model_label: bare id for N=1, "majority(n=K): ..." sorted for N>1.
  - Panel records one vote per judge per candidate with distinct mock labels.

Run: PYTHONPATH=scripts .venv/bin/python scripts/lookalike/tests/test_judge_panel.py
"""
from __future__ import annotations

import sys

from lookalike.common import Candidate, JudgeVote, RunResult, Seed
from lookalike.judge import (
    JudgePanel,
    _aggregate_votes,
    _mock_score,
    judge_model_label,
)

SEED = Seed(seed_slug="pylon", seed_name="Pylon", seed_domain="usepylon.com",
            description="B2B support", category="b2b-saas")


def _run() -> RunResult:
    cands = [Candidate(name=f"Co{i}", domain=f"co{i}.com", rank=i + 1) for i in range(8)]
    return RunResult(seed_slug="pylon", provider_slug="x", config_name="c", config={}, candidates=cands, latency_ms=1)


def _votes(bools: list[bool]) -> list[JudgeVote]:
    return [JudgeVote(judge_model=f"m{i}", relevant=b, rationale="r") for i, b in enumerate(bools)]


def main() -> int:
    f: list[str] = []

    # label builder
    if judge_model_label(["gpt-5.4-mini"]) != "gpt-5.4-mini":
        f.append("label N=1 wrong")
    if judge_model_label(["b", "a", "a"]) != "majority(n=2): a,b":
        f.append(f"label N>1 wrong: {judge_model_label(['b','a','a'])}")

    # aggregation
    cases = [
        ([True], True), ([False], False),
        ([True, False], False),            # even-N tie -> False
        ([True, True], True),
        ([True, False, True], True),
        ([True, False, False], False),
        ([True, True, False, False], False),  # 2/4 tie -> False
        ([True, True, True, False], True),
    ]
    for bools, expected in cases:
        rel, _ = _aggregate_votes(_votes(bools))
        if rel != expected:
            f.append(f"aggregate {bools} -> {rel}, expected {expected}")

    # N=1 mock byte-identical to legacy _mock_score
    panel1 = JudgePanel(models=["mock"], mock=True)
    if panel1.label() != "mock-judge":
        f.append(f"panel N=1 mock label {panel1.label()!r} != 'mock-judge'")
    jr1 = panel1.score_run(SEED, _run())
    for jc in jr1.judged:
        legacy = _mock_score(SEED, jc.candidate, "", 0.6)
        if jc.relevant != legacy.relevant or jc.rationale != legacy.rationale:
            f.append(f"N=1 mock diverged on {jc.candidate.name}: {jc.relevant}/{jc.rationale!r} vs {legacy.relevant}/{legacy.rationale!r}")
        if len(jc.votes) != 1 or jc.votes[0].judge_model != "mock-judge":
            f.append(f"N=1 vote shape wrong on {jc.candidate.name}: {jc.votes}")

    # N=3 mock: 3 votes/candidate, distinct labels, relevant == majority of the 3
    panel3 = JudgePanel(models=["a", "b", "c"], mock=True)
    if panel3.label() != "mock-judge":
        f.append("panel N=3 mock label not 'mock-judge'")
    jr3 = panel3.score_run(SEED, _run())
    salts = [("", 0.6), ("mock-1", 0.55), ("mock-2", 0.65)]
    saw_disagreement = False
    for jc in jr3.judged:
        if [v.judge_model for v in jc.votes] != ["mock-judge", "mock-judge-1", "mock-judge-2"]:
            f.append(f"N=3 vote labels wrong on {jc.candidate.name}: {[v.judge_model for v in jc.votes]}")
        indiv = [_mock_score(SEED, jc.candidate, s, t).relevant for s, t in salts]
        if len(set(indiv)) > 1:
            saw_disagreement = True
        expected = sum(indiv) > len(indiv) / 2
        if jc.relevant != expected:
            f.append(f"N=3 majority wrong on {jc.candidate.name}: {jc.relevant} vs {expected} (votes {indiv})")
    if not saw_disagreement:
        f.append("N=3 mock judges never disagreed across 8 candidates — salts not varying")

    # --- concurrent (live-path) judging: every judge call is still captured in
    # the audit buffer across worker threads, and aggregation is unchanged ---
    from collections import Counter

    from lookalike.common import JudgedCandidate
    from lookalike.judge import _emit_judge_trace, capture_judge_calls

    class _FakeJudge:  # non-network judge that emits a trace like the real one
        def __init__(self, model: str) -> None:
            self.model = model

        def score_candidate(self, seed: Seed, c: Candidate) -> JudgedCandidate:
            _emit_judge_trace(
                model=self.model, messages=[{"role": "user", "content": c.name}],
                raw_response="{}", parsed={"relevant": True, "rationale": "r"},
                elapsed_ms=1, for_candidate_rank=c.rank,
            )
            return JudgedCandidate(candidate=c, relevant=True, rationale="r")

    panelc = JudgePanel(models=["a", "b"], mock=True)  # built without creds
    panelc.mock = False          # force the threaded live path in score_run
    panelc.concurrency = 4
    panelc.judges = [_FakeJudge("a"), _FakeJudge("b")]  # type: ignore[list-item]
    panelc.vote_labels = ["a", "b"]
    runc = RunResult(
        seed_slug="pylon", provider_slug="x", config_name="c", config={},
        candidates=[Candidate(name=f"Co{i}", domain=f"co{i}.com", rank=i + 1) for i in range(10)],
        latency_ms=1,
    )
    with capture_judge_calls() as buf:
        jrc = panelc.score_run(SEED, runc)
    if len(buf) != 20:
        f.append(f"concurrent capture: got {len(buf)} judge calls, expected 20 (copy_context not propagating to workers)")
    per_rank = Counter(call.for_candidate_rank for call in buf)
    if any(per_rank[r] != 2 for r in range(1, 11)):
        f.append(f"concurrent capture: per-candidate trace counts off: {dict(per_rank)}")
    if len(jrc.judged) != 10 or not all(len(jc.votes) == 2 for jc in jrc.judged):
        f.append("concurrent score_run: wrong judged/votes shape")
    if not all(jc.relevant for jc in jrc.judged):
        f.append("concurrent score_run: both-yes should aggregate to relevant=True")

    if f:
        print(f"{len(f)} FAILURE(s):")
        for x in f:
            print("  - " + x)
        return 1
    print("PASS — N=1 mock byte-identical; majority/tie rules correct; labels + vote structure correct; panel disagrees when expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
