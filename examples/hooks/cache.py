"""Cache decisions, and skip inference on a hit.

A start hook calls `ctx.skip(results)`; the engine skips the forward pass and still runs
the end hooks. The key has to cover everything the answer depends on, so it takes the whole
context rather than just the state and the questions.

    python examples/hooks/cache.py
"""
import hashlib
import json

import laya

CACHE = {}


def cache_key(ctx):
    """One entry per question the model is actually asked.

    Deliberately not `sort_keys=True`: a choice question's criteria order is positional, so two
    orders are two questions, and `_question_schema` in `laya/router.py` keeps them apart for the
    same reason. Sorting the keys folds them into one entry, and the second caller gets the first
    caller's numbers. `ctx.model` and the token budget belong in the key for the same reason: on
    the Router one hook set serves three checkpoints, and a smaller `max_len` truncates the state.
    """
    payload = json.dumps([ctx.states[0], ctx.questions, ctx.model,
                          ctx.max_len, ctx.head_max_len], default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def cache_read(ctx):
    hit = CACHE.get(cache_key(ctx))
    if hit is not None:
        ctx.skip([hit])


def cache_write(ctx):
    if ctx.results:
        CACHE[cache_key(ctx)] = ctx.results[0]


agent = laya.load("convaiinnovations/laya",
                  on_predict_start=cache_read, on_predict_end=cache_write)

STATE = "I was charged twice for the same invoice."
CRITERIA = {"refund": "give me money back", "cancel": "stop the service",
            "other": "anything else"}
QUESTIONS = {"ask": {"type": "choice", "instructions": "What does the customer want?",
                     "criteria": dict(CRITERIA)}}
# Same labels and descriptions, criteria written in the other order.
REORDERED = {"ask": {"type": "choice", "instructions": "What does the customer want?",
                     "criteria": {"other": CRITERIA["other"], "refund": CRITERIA["refund"],
                                  "cancel": CRITERIA["cancel"]}}}

first = agent.system_one(STATE, QUESTIONS)     # runs the model, fills the cache
again = agent.system_one(STATE, QUESTIONS)     # served from CACHE, no forward pass
agent.system_one(STATE, REORDERED)             # runs too: a reordered question is a new entry
print("cache entries:", len(CACHE))            # 2, not 1
print("same answer:", first["answers"] == again["answers"])
