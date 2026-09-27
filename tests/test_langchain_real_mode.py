"""The shape `pip install "laya[langchain]"` actually produces.

`tests/test_langchain.py` is listed in the `tests` CI job, and that job installs `-e ".[mcp,serve]"`.
Neither extra is `langchain`, so `laya/integrations/langchain.py` takes its `except ImportError` branch
and sets `RunnableSerializable = object`; all 31 of that file's checks then run against plain Python
objects. Measured both ways, the output is identical -- `PASS: 31`, `FAIL: 0`, exit 0 -- so the job
cannot tell the two environments apart even though they disagree about what a router does:

    confidence_threshold='0.8'   real: 0.8 (coerced)   shim: '0.8' (kept as a str)
    .invoke(), confidence 0.5    real: 'human'         shim: TypeError: '>' not supported
                                                        between instances of 'str' and 'float'
    .batch([two inputs])         real: two labels      shim: AttributeError: no attribute 'batch'

This file asserts the real column. It fails rather than skipping when langchain-core is absent: a skip
is the failure mode above, and the job's own comment for installing `serve` says it is there so that
suites do not "skip themselves silently and pass".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.integrations.langchain import (
    LayaEvaluator,
    LayaGuardrail,
    LayaRouter,
    LayaTriage,
    RunnableSerializable,
    _RUNNABLE_AVAILABLE,
)

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append(f"{name}:\n     got  {got!r}\n     want {want!r}")


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append(f"{name} {detail}")


def check_raises(name, fn, kind):
    try:
        value = fn()
    except Exception as error:
        check_true(name, isinstance(error, kind),
                   f"(raised {type(error).__name__}: {error}, wanted {kind.__name__})")
        return
    FAIL.append(f"{name}: accepted {value!r}, wanted {kind.__name__}")


def outcome(fn):
    """Report a failure instead of raising out of the suite: the fallback base class turns a
    coerced threshold into a TypeError at decision time, and that is a result to print."""
    try:
        return fn()
    except Exception as error:
        return f"{type(error).__name__}: {error}"


if not _RUNNABLE_AVAILABLE or RunnableSerializable is object:
    print("FAIL: 1")
    print("FAILED: langchain-core is not installed, so laya.integrations.langchain fell back to "
          "`RunnableSerializable = object`. Install the extra this suite is for: "
          "pip install -e \".[langchain]\" (docs/langchain.md: pip install \"laya[langchain]\")")
    sys.exit(1)

from pydantic import ValidationError  # noqa: E402  # the base class is a pydantic model


# --------------------------------------------------------------- Mock Agent
class MockAgent:
    """Answers every question with the same choice at a fixed confidence, no weights."""

    def __init__(self, confidence):
        self.confidence = confidence

    def predict(self, state, questions, **kwargs):
        return {"answers": {qid: {"choice": "billing_agent", "confidence": self.confidence,
                                  "urgency": 1, "needs_human": False}
                            for qid in questions}}


low, high = MockAgent(0.5), MockAgent(0.9)
CRITERIA = {"billing_agent": "invoices, refunds, billing", "tech": "bugs, crashes, errors"}

# --------------------------------------------------------------- 1. The base class is real
for cls in (LayaRouter, LayaGuardrail, LayaTriage, LayaEvaluator):
    check_true(f"base/{cls.__name__}", issubclass(cls, RunnableSerializable),
               f"(mro is {cls.__mro__[1].__name__}, not RunnableSerializable)")
    check_true(f"runnable-api/{cls.__name__}", hasattr(cls, "batch"),
               "(a Runnable gets .batch() from its base; the shim has no such method)")

# --------------------------------------------------------------- 2. Fields validate and coerce
ROUTER_KW = dict(criteria=CRITERIA, confidence_threshold="0.8", fallback="human")
router = outcome(lambda: LayaRouter(agent=low, **ROUTER_KW))
check_true("construct/accepts-a-string-threshold", not isinstance(router, str),
           f"(got {router!r})")
check("coerce/threshold-type", outcome(lambda: type(router.confidence_threshold).__name__), "float")
check("coerce/threshold-value", outcome(lambda: router.confidence_threshold), 0.8)
check_raises("validate/criteria-int",
             lambda: LayaRouter(criteria={"billing_agent": 1}, agent=high), ValidationError)
check_raises("validate/instructions-none",
             lambda: LayaRouter(criteria=CRITERIA, instructions=None, agent=high), ValidationError)

# --------------------------------------------------------------- 3. And the coerced value decides
check("gate/below-threshold", outcome(lambda: router.invoke({"input": "I was billed twice"})),
      "human")
check("gate/above-threshold",
      outcome(lambda: LayaRouter(criteria=CRITERIA, confidence_threshold="0.8", fallback="human",
                                 agent=high).invoke({"input": "I was billed twice"})),
      "billing_agent")
check("gate/numeric-threshold-still-gates",
      outcome(lambda: LayaRouter(criteria=CRITERIA, confidence_threshold=0.8, fallback="human",
                                 agent=low).invoke({"input": "I was billed twice"})),
      "human")

# --------------------------------------------------------------- 4. LCEL surface
batched = outcome(lambda: LayaRouter(criteria=CRITERIA, agent=high).batch(
    ["I was billed twice", "the app crashes on launch"]))
check_true("batch/returns-one-result-per-input", isinstance(batched, list) and len(batched) == 2,
           f"(got {batched!r})")
check_true("batch/results-are-branch-labels",
           isinstance(batched, list) and all(isinstance(r, str) for r in batched),
           f"(got {batched!r})")

# --------------------------------------------------------------- Summary
print(f"PASS: {len(PASS)}")
print(f"FAIL: {len(FAIL)}")
for f in FAIL:
    print(f"FAILED: {f}")

if FAIL:
    sys.exit(1)
