"""API-stability guard for prediction hooks.

These tests pin the public hook surface (parameter names, kinds, defaults, context fields,
lifecycle events, exports) so a change that would break callers fails here first. If a change
is intentional, update this file in the same commit.

The cache key the examples and docs teach is pinned here too: `ctx.skip()` hands back whatever
the key matched, so what a key covers is part of the contract, not an implementation detail.

Run: python tests/test_hooks_api.py
"""
import dataclasses
import hashlib
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import laya  # noqa: E402
from laya import Agent, AsyncHook, BaseHook, PredictContext, PredictHook, Router, load  # noqa: E402
from laya.hooks import HOOK_EVENTS, Hook  # noqa: E402
from laya.onnx_agent import ONNXAgent  # noqa: E402
from laya.router import _question_schema  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


def sig(fn):
    return inspect.signature(fn).parameters


def check_param(name, fn, param, default, kind=None):
    params = sig(fn)
    if param not in params:
        FAIL.append("%s/%s: missing parameter" % (name, param))
        return
    p = params[param]
    check("%s/%s default" % (name, param), p.default, default)
    if kind is not None:
        check("%s/%s kind" % (name, param), p.kind, kind)


HOOK_KEYS = {
    "hooks": None,
    "on_predict_start": None,
    "on_predict_end": None,
    "hooks_raise": True,
    "hooks_concurrent": True,
    "hooks_timeout": None,
}

# --------------------------------------------------------------- constructors
for label, fn in (("Agent.__init__", Agent.__init__), ("load", load),
                  ("Router.__init__", Router.__init__), ("ONNXAgent.__init__", ONNXAgent.__init__)):
    for param, default in HOOK_KEYS.items():
        check_param(label, fn, param, default)

# Router keeps lang_guess and explicit per-model revisions too
check_param("Router.__init__", Router.__init__, "lang_guess", None)
check_param("Router.__init__", Router.__init__, "revisions", None)

# --------------------------------------------------------------- predict surfaces
for label, fn in (("Agent.predict_batch", Agent.predict_batch),
                  ("Agent.system_one", Agent.system_one),
                  ("Router.predict", Router.predict),
                  ("ONNXAgent.system_one", ONNXAgent.system_one)):
    check_param(label, fn, "hooks", None)
    check_param(label, fn, "on_predict_start", None)
    check_param(label, fn, "on_predict_end", None)
    check_param(label, fn, "hooks_raise", None)
    check_param(label, fn, "hooks_timeout", None)

check_param("Agent.predict_batch", Agent.predict_batch, "batch_size", None)

# per-call token-budget overrides
for label, fn in (("Agent.predict_batch", Agent.predict_batch),
                  ("Agent.system_one", Agent.system_one),
                  ("Router.predict", Router.predict),
                  ("ONNXAgent.system_one", ONNXAgent.system_one)):
    check_param(label, fn, "max_len", None)
    check_param(label, fn, "head_max_len", None)

# route() takes per-call hooks so a hook can pin a checkpoint for one call
check_param("Router.route", Router.route, "hooks", None)
check_param("Router.route", Router.route, "hooks_raise", None)
check_param("Router.route", Router.route, "hooks_timeout", None)

# --------------------------------------------------------------- aliases
check_true("Agent.predict is Agent.system_one", Agent.predict is Agent.system_one)
check_true("Router.system_one is Router.predict", Router.system_one is Router.predict)
check_true("ONNXAgent.predict is ONNXAgent.system_one", ONNXAgent.predict is ONNXAgent.system_one)

# --------------------------------------------------------------- context
FIELDS = ["states", "questions", "run_id", "results", "decision", "model", "agent", "router",
          "max_len", "head_max_len", "usage", "started_at", "elapsed_ms", "error"]
check("PredictContext fields", [f.name for f in dataclasses.fields(PredictContext)], FIELDS)
check("PredictContext/states required", PredictContext.__dataclass_fields__["states"].default,
      dataclasses.MISSING)
check("PredictContext/questions required", PredictContext.__dataclass_fields__["questions"].default,
      dataclasses.MISSING)
for optional in ("results", "decision", "model", "agent", "router", "usage", "elapsed_ms", "error"):
    check("PredictContext/%s default None" % optional,
          PredictContext.__dataclass_fields__[optional].default, None)
check_true("PredictContext/skip exists", callable(getattr(PredictContext, "skip", None)))
check_true("PredictContext/run_id has a factory",
           PredictContext.__dataclass_fields__["run_id"].default_factory is not dataclasses.MISSING)

# --------------------------------------------------------------- hook protocol
check("Hook lifecycle events", set(HOOK_EVENTS),
      {"on_predict_start", "on_predict_end", "on_route", "on_load", "on_evict", "on_error"})
for event in HOOK_EVENTS:
    check_true("Hook/%s declared" % event, hasattr(Hook, event))
check_true("PredictHook is callable-typed", callable(PredictHook))

# --------------------------------------------------------------- exports
for name in ("PredictContext", "PredictHook", "Hook", "BaseHook", "AsyncHook"):
    check_true("__all__/%s" % name, name in laya.__all__)
    check_true("laya.%s exists" % name, hasattr(laya, name))
check_true("laya.hooks/run_coroutine_sync exists",
           callable(getattr(__import__("laya.hooks", fromlist=["run_coroutine_sync"]),
                            "run_coroutine_sync", None)))

# BaseHook is the concrete no-op base class; all six events exist and are callable.
for event in HOOK_EVENTS:
    check_true("BaseHook/%s callable" % event, callable(getattr(BaseHook, event, None)))

# process-wide default registry lives in laya.hooks (not the top level)
for helper in ("default_hooks", "set_default_hooks", "add_default_hook", "clear_default_hooks",
               "compose_hooks"):
    check_true("laya.hooks/%s exists" % helper, callable(getattr(__import__("laya.hooks", fromlist=[helper]), helper, None)))
check_true("defaults/not exported at top level", not hasattr(laya, "set_default_hooks"))

# --------------------------------------------------------------- class defaults
for label, cls in (("Agent", Agent), ("Router", Router), ("ONNXAgent", ONNXAgent)):
    check("%s/hooks default" % label, cls.hooks, ())
    check("%s/hooks_raise default" % label, cls.hooks_raise, True)
    check("%s/hooks_concurrent default" % label, cls.hooks_concurrent, True)
    check("%s/hooks_timeout default" % label, cls.hooks_timeout, None)
    check("%s/_hooks_lock default" % label, cls._hooks_lock, None)

# Agent and ONNXAgent carry the checkpoint id for ctx.model; Router has no single model.
for label, cls in (("Agent", Agent), ("ONNXAgent", ONNXAgent)):
    check("%s/model_id default" % label, cls.model_id, None)

# runtime registration surface
for label, cls in (("Agent", Agent), ("Router", Router), ("ONNXAgent", ONNXAgent)):
    for method in ("add_hook", "remove_hook", "hooks_installed"):
        check_true("%s/%s exists" % (label, method), callable(getattr(cls, method, None)))


# ------------------------------------------------- the cache key the examples and docs teach
# `examples/hooks/cache.py`, and the caching blocks of `docs/hooks/patterns.md` and
# `docs/hooks/examples.md`, are three copies of one `ctx.skip()` pattern. The key is the part that
# can be wrong: a payload sorted by key folds two criteria orders into one cache entry, while
# `_question_schema` keeps them apart because a choice question's option order is positional. The
# second caller then gets the first caller's answer. Measured on the English checkpoint, a question
# whose criteria were only reordered came back at 0.661 confidence from the cache instead of its
# own 0.496 -- enough to open a gate at 0.6 that a fresh pass would have kept shut.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CRITERIA = {"refund": "give me money back", "cancel": "stop the service", "other": "anything else"}
Q_ORDER_A = {"ask": {"type": "choice", "instructions": "What does the customer want?",
                     "criteria": dict(CRITERIA)}}
Q_ORDER_B = {"ask": {"type": "choice", "instructions": "What does the customer want?",
                     "criteria": {"other": CRITERIA["other"], "refund": CRITERIA["refund"],
                                  "cancel": CRITERIA["cancel"]}}}


def taught_key(path, marker):
    """The key builder of one of those three files, exec'd out of that file's own text.

    Sliced rather than imported: the example calls `laya.load()` at module scope, which would
    download a checkpoint, and a docs code block is not importable at all.
    """
    with open(os.path.join(REPO, path), encoding="utf-8") as handle:
        text = handle.read()
    start = text.index("def %s(" % marker)
    namespace = {"json": json, "hashlib": hashlib}
    exec(text[start:text.index("\ndef ", start + 1)], namespace)
    return namespace[marker]


def cache_ctx(questions, model="english", max_len=None, head_max_len=None):
    return PredictContext(states=["I was charged twice for the same invoice."],
                          questions=questions, model=model, max_len=max_len,
                          head_max_len=head_max_len)


TAUGHT = [("examples/hooks/cache.py", taught_key("examples/hooks/cache.py", "cache_key")),
          ("docs/hooks/patterns.md", taught_key("docs/hooks/patterns.md", "key")),
          ("docs/hooks/examples.md", taught_key("docs/hooks/examples.md", "key"))]

for path, key in TAUGHT:
    ctx = cache_ctx(Q_ORDER_A)
    check_true("%s/reordered criteria is a new entry" % path, key(ctx) != key(cache_ctx(Q_ORDER_B)))
    check_true("%s/reorders are distinct exactly when the core says so" % path,
               (key(ctx) != key(cache_ctx(Q_ORDER_B)))
               == (_question_schema(Q_ORDER_A) != _question_schema(Q_ORDER_B)))
    check_true("%s/the same call is a hit" % path, key(ctx) == key(cache_ctx(Q_ORDER_A)))
    check_true("%s/another checkpoint is a new entry" % path,
               key(ctx) != key(cache_ctx(Q_ORDER_A, model="multilingual")))
    for field in ("max_len", "head_max_len"):
        check_true("%s/another %s is a new entry" % (path, field),
                   key(ctx) != key(cache_ctx(Q_ORDER_A, **{field: 256})))

# Three copies, one key: a docs page that drifts from the example fails here, not in someone's
# production cache.
for path, key in TAUGHT[1:]:
    check("%s/keyed like the example" % path, key(cache_ctx(Q_ORDER_A)), TAUGHT[0][1](cache_ctx(Q_ORDER_A)))


print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL", f)
if not FAIL:
    print("all hook API tests passed")
sys.exit(1 if FAIL else 0)
