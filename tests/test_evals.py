"""Metric math, dataset parsing and regression comparison for laya.evals. No weights.

Run: python -m pytest tests/test_evals.py -q
"""
import json

import pytest

from laya.evals import (
    ChoiceAccuracy,
    Dataset,
    EvalError,
    EvalReport,
    Example,
    MeanConfidence,
    NoulAccuracy,
    ScoreMAE,
    ScoreWithin,
    assert_regression,
    default_evaluators,
    ece,
    evaluate,
)

Q = {"intent": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}}}
QSCORE = {"quality": {"type": "score", "instructions": "?", "criteria": ["low", "high"]}}
QNOUL = {"flag": {"type": "noul", "instructions": "?"}}


def choice_answer(label, confidence=0.9):
    return {"type": "choice", "choice": label, "probabilities": {label: confidence},
            "confidence": confidence}


def noul_answer(prob):
    return {"type": "noul", "noul": prob, "confidence": max(prob, 1 - prob)}


class StubRunner:
    """Returns fixed answers per state, so evaluation is deterministic and weight-free."""

    def __init__(self, by_state):
        self.by_state = by_state

    def predict(self, state, questions, model=None):
        return {"model": model or "stub", "answers": self.by_state[state]}


# --------------------------------------------------------------- evaluator math
def test_choice_and_confidence_math():
    evaluator = ChoiceAccuracy()
    assert evaluator.score(choice_answer("a"), "a") == 1.0
    assert evaluator.score(choice_answer("b"), "a") == 0.0
    assert evaluator.score(noul_answer(0.9), "a") is None, "wrong answer type does not apply"
    assert MeanConfidence().score(choice_answer("a", 0.8), "a") == 0.8


def test_noul_and_score_math():
    assert NoulAccuracy().score(noul_answer(0.9), True) == 1.0
    assert NoulAccuracy().score(noul_answer(0.2), True) == 0.0
    assert ScoreMAE().score({"type": "score", "score": 0.4}, 0.7) == pytest.approx(0.3)
    within = ScoreWithin(0.5)
    assert within.score({"type": "score", "score": 0.4}, 0.7) == 1.0
    assert within.name == "score_within_0.5"


def test_calibration_uses_answer_confidence():
    answer = {"type": "choice", "choice": "a", "confidence": 0.2, "answer_confidence": 0.9}
    assert MeanConfidence().score(answer, "a") == pytest.approx(0.9), "calibrated, not entropy"
    report = evaluate(StubRunner({"s": {"intent": answer}}),
                      Dataset([Example("s", Q, {"intent": "a"})]))
    assert report.overall["mean_confidence"] == pytest.approx(0.9)


def test_compare_ignores_latency_by_default():
    report = EvalReport(overall={"choice_accuracy": 0.8, "latency_p50_ms": 12.0})
    baseline = {"overall": {"choice_accuracy": 0.8, "latency_p50_ms": 5.0}}
    ok, deltas = report.compare(baseline)
    assert ok and "latency_p50_ms" not in deltas, "timing noise is not a quality regression"
    bad, deltas = report.compare(baseline, {"latency_p50_ms": 1.0})
    assert not bad and "latency_p50_ms" in deltas


def test_ece_on_known_inputs():
    assert ece([1.0, 1.0], [True, False]) == pytest.approx(0.5)
    assert ece([0.0, 0.0], [False, False]) == pytest.approx(0.0)
    assert ece([], []) is None


# --------------------------------------------------------------- dataset
def test_dataset_from_jsonl(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join([
        json.dumps({"state": "s1", "questions": Q, "expected": {"intent": "a"}, "language": "en"}),
        "# a comment line",
        json.dumps({"state": "s2", "questions": Q, "expected": {"intent": "b"}, "tags": ["t"]}),
    ]), encoding="utf-8")
    dataset = Dataset.from_jsonl(str(path))
    assert len(dataset) == 2
    assert dataset.examples[0].language == "en"
    assert dataset.examples[1].tags == ("t",)


@pytest.mark.parametrize("row, fragment", [
    ({"state": "s", "questions": Q}, "missing 'expected'"),
    ({"state": "s", "questions": Q, "expected": {"nope": "a"}}, "unknown question"),
    ({"state": "s", "questions": [], "expected": {}}, "'questions' must be an object"),
])
def test_dataset_rejects_bad_rows(row, fragment):
    with pytest.raises(EvalError) as exc:
        Example.from_dict(row)
    assert fragment in str(exc.value)


def test_dataset_rejects_malformed_json_and_empty(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(EvalError):
        Dataset.from_jsonl(str(bad))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(EvalError):
        Dataset.from_jsonl(str(empty))


# --------------------------------------------------------------- evaluate
def test_evaluate_overall_and_slices():
    dataset = Dataset([
        Example("s1", Q, {"intent": "a"}, language="en"),
        Example("s2", Q, {"intent": "b"}, language="en"),
        Example("s3", Q, {"intent": "a"}, language="de"),
    ])
    runner = StubRunner({
        "s1": {"intent": {"type": "choice", "choice": "a", "confidence": 1.0}},
        "s2": {"intent": {"type": "choice", "choice": "b", "confidence": 1.0}},
        "s3": {"intent": {"type": "choice", "choice": "b", "confidence": 1.0}},
    })
    report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()])
    assert report.overall["choice_accuracy"] == pytest.approx(2 / 3)
    assert report.slices["language"]["en"]["choice_accuracy"] == pytest.approx(1.0)
    assert report.slices["language"]["de"]["choice_accuracy"] == pytest.approx(0.0)
    assert report.slices["qid"]["intent"]["choice_accuracy"] == pytest.approx(2 / 3)
    assert "choice_accuracy" in report.to_markdown()


def test_evaluate_batches_same_questions():
    class BatchRunner(StubRunner):
        def __init__(self, by_state):
            super().__init__(by_state)
            self.batches = []

        def predict_batch(self, states, questions, model=None, batch_size=None):
            self.batches.append(list(states))
            return [{"model": "m", "answers": self.by_state[s]} for s in states]

    dataset = Dataset([Example("s1", Q, {"intent": "a"}), Example("s2", Q, {"intent": "a"})])
    runner = BatchRunner({"s1": {"intent": choice_answer("a")}, "s2": {"intent": choice_answer("a")}})
    report = evaluate(runner, dataset, evaluators=[ChoiceAccuracy()], batch_size=8)
    assert runner.batches == [["s1", "s2"]], "identical questions share one forward pass"
    assert report.overall["choice_accuracy"] == 1.0


def test_evaluate_skips_errors_when_asked():
    class Boom(StubRunner):
        def predict(self, state, questions, model=None):
            if state == "bad":
                raise RuntimeError("no model")
            return super().predict(state, questions, model)

    dataset = Dataset([Example("ok", Q, {"intent": "a"}), Example("bad", Q, {"intent": "a"})])
    runner = Boom({"ok": {"intent": choice_answer("a")}})
    with pytest.raises(RuntimeError):
        evaluate(runner, dataset, on_error="fail")
    report = evaluate(runner, dataset, on_error="skip")
    assert len(report.cases) == 1
    assert report.config["errored"][0]["error"].startswith("RuntimeError")


# --------------------------------------------------------------- compare
def test_compare_and_assert_regression():
    report = EvalReport(overall={"choice_accuracy": 0.8, "ece": 0.10})
    baseline = {"overall": {"choice_accuracy": 0.79, "ece": 0.08}}
    ok, deltas = report.compare(baseline, {"choice_accuracy": 0.02, "ece": 0.03})
    assert ok and deltas["choice_accuracy"]["diff"] == pytest.approx(0.01)
    bad, bad_deltas = report.compare(baseline, {"ece": 0.01})
    assert not bad and bad_deltas["ece"]["diff"] == pytest.approx(0.02)
    with pytest.raises(AssertionError):
        assert_regression(report, baseline, {"ece": 0.01})


def test_default_evaluators_cover_the_three_types():
    names = {e.name for e in default_evaluators()}
    assert {"choice_accuracy", "noul_accuracy", "score_mae", "mean_confidence"} <= names


# --------------------------------------------------------------- CLI
def _write_dataset(tmp_path, rows):
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def test_cli_validate_and_dispatch(tmp_path):
    from laya import cli, evals_cli

    good = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    assert evals_cli.main(["validate", good]) == 0
    assert cli.main(["eval", "validate", good]) == 0, "`laya eval` dispatches to laya-evals"

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    assert evals_cli.main(["validate", str(bad)]) == 1


def test_cli_compare_exit_codes(tmp_path):
    from laya import evals_cli

    (tmp_path / "report.json").write_text(json.dumps({"overall": {"choice_accuracy": 0.80}}))
    (tmp_path / "baseline.json").write_text(json.dumps({"overall": {"choice_accuracy": 0.79}}))
    report, baseline = str(tmp_path / "report.json"), str(tmp_path / "baseline.json")
    assert evals_cli.main(["compare", report, "--baseline", baseline,
                           "--tolerance", "choice_accuracy=0.02"]) == 0
    assert evals_cli.main(["compare", report, "--baseline", baseline,
                           "--tolerance", "choice_accuracy=0.001"]) == 1


def test_cli_rejects_a_malformed_tolerance():
    from laya import evals_cli

    with pytest.raises(EvalError):
        evals_cli._parse_pairs(["choice_accuracy"])


# --------------------------------------------------------------- --revision pinning
SHA = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


def _fake_router(monkeypatch, recorded):
    """Replace `laya.Router` with a weight-free stand-in that records how `_cmd_run` built it.

    It validates checkpoint names the way `Router.__init__` does, by calling the same
    `normalise_name`, so the fake refuses a typo for the real reason.
    """
    import laya

    class FakeRouter:
        def __init__(self, device=None, preload=None, revision=None, revisions=None):
            from laya.router import normalise_name
            recorded.update(device=device, preload=preload, revision=revision,
                            revisions={normalise_name(k): v for k, v in (revisions or {}).items()})

        def predict(self, state, questions, model=None):
            return {"model": model or "english", "answers": {"intent": choice_answer("a")}}

        loaded_revisions = {"english": SHA}

    monkeypatch.setattr(laya, "Router", FakeRouter)


def test_cli_forwards_the_pin_and_records_the_commit_that_answered(tmp_path, monkeypatch):
    from laya import evals_cli

    recorded = {}
    _fake_router(monkeypatch, recorded)
    dataset = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    out = tmp_path / "report.json"
    assert evals_cli.main(["run", dataset, "--model", "english", "--device", "cpu",
                           "--revision", "english=" + SHA, "--json", str(out)]) == 0
    assert recorded["revision"] is None
    assert recorded["revisions"] == {"english": SHA}
    # The report is the artifact a reviewer commits, so it has to carry the commit itself.
    assert json.loads(out.read_text())["config"]["revisions"] == {"english": SHA}


def test_cli_bare_revision_pins_every_checkpoint(tmp_path, monkeypatch):
    from laya import evals_cli

    recorded = {}
    _fake_router(monkeypatch, recorded)
    dataset = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    assert evals_cli.main(["run", dataset, "--revision", SHA, "--json",
                           str(tmp_path / "r.json")]) == 0
    # A bare SHA is one commit for every checkpoint, so nothing is pinned per name.
    assert recorded["revision"] == SHA and recorded["revisions"] == {}


def test_cli_records_an_unpinned_run_too(tmp_path, monkeypatch):
    """The default branch is what an unpinned baseline was taken on; the report must say so."""
    from laya import evals_cli

    recorded = {}
    _fake_router(monkeypatch, recorded)
    dataset = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    out = tmp_path / "r.json"
    assert evals_cli.main(["run", dataset, "--json", str(out)]) == 0
    assert recorded["revision"] is None and recorded["revisions"] == {}
    assert json.loads(out.read_text())["config"]["revisions"] == {"english": SHA}


def test_cli_rejects_a_typo_in_a_pinned_checkpoint_name(tmp_path, monkeypatch, capsys):
    from laya import evals_cli

    recorded = {}
    _fake_router(monkeypatch, recorded)
    dataset = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    assert evals_cli.main(["run", dataset, "--revision", "englishg=" + SHA]) == 1
    err = capsys.readouterr().err
    assert "unknown model 'englishg'" in err
    assert "choose one of" in err, "the message comes from core, with the option list"
    assert "Traceback" not in err, "a mistyped pin is a usage error, not a crash"
    assert not recorded, "and it fails before any checkpoint is loaded"


def test_cli_accepts_a_checkpoint_alias_in_a_pin(tmp_path, monkeypatch):
    """`Router` owns the alias table, so `en=` must reach it rather than be second-guessed here."""
    from laya import evals_cli

    recorded = {}
    _fake_router(monkeypatch, recorded)
    dataset = _write_dataset(tmp_path, [{"state": "s", "questions": Q, "expected": {"intent": "a"}}])
    assert evals_cli.main(["run", dataset, "--revision", "en=" + SHA]) == 0
    assert recorded["revisions"] == {"english": SHA}


@pytest.mark.parametrize("pairs, expected", [
    (None, (None, {})),
    ([], (None, {})),
    ([SHA], (SHA, {})),
    ([SHA, SHA], (SHA, {})),                      # repeating one commit is agreement, not a clash
    (["en=" + SHA], (None, {"en": SHA})),
    (["en=" + SHA, SHA], (SHA, {"en": SHA})),     # both forms at once: Router lets the pair win
])
def test_parse_revisions_accepts_both_forms(pairs, expected):
    from laya import evals_cli

    assert evals_cli._parse_revisions(pairs) == expected


@pytest.mark.parametrize("pair, fragment", [
    ("=abc", "NAME=REVISION"),
    ("en=", "NAME=REVISION"),
    ("   ", "commit SHA"),
    ("abc", None),
])
def test_parse_revisions_rejects_half_a_pair(pair, fragment):
    from laya import evals_cli

    if fragment is None:
        assert evals_cli._parse_revisions([pair]) == (pair, {})
        return
    with pytest.raises(EvalError) as exc:
        evals_cli._parse_revisions([pair])
    assert fragment in str(exc.value)


def test_parse_revisions_rejects_two_different_bare_commits():
    from laya import evals_cli

    with pytest.raises(EvalError) as exc:
        evals_cli._parse_revisions(["abc", "def"])
    assert "two commits" in str(exc.value)
