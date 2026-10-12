"""Contract guard for laya.evidence: readiness evidence states over existing artifacts.

Run: python tests/test_evidence.py
"""
import inspect
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya import evidence  # noqa: E402
from laya import evals_cli  # noqa: E402

PASS, FAIL = [], []


def check(name, got, expected):
    ok = got == expected
    (PASS if ok else FAIL).append(name)
    print(("  ok   " if ok else "  FAIL ") + name, "" if ok else "(got %r, expected %r)" % (got, expected))


def check_true(name, condition):
    check(name, bool(condition), True)


def calib(config):
    return evidence.inspect_calibration(config)


# ------------------------------------------------------ calibration states
legacy = calib({"fine_tuned": True})
check("legacy checkpoint (no training key) -> UNKNOWN", legacy["state"], evidence.UNKNOWN)

legacy2 = calib({"training": {"laya_train": {"base": "x"}}})
check("training but no calibration block -> UNKNOWN", legacy2["state"], evidence.UNKNOWN)

zero = calib({"training": {"laya_train_calibration": {"choice": {"items": 0, "temperature": 1.0, "issues": []}}}})
check("zero items -> MISSING", zero["types"]["choice"]["state"], evidence.MISSING)

low = calib({"training": {"laya_train_calibration": {"choice": {"items": 5, "temperature": 1.0,
        "issues": ["not fitted: 5 calibration items, fewer than 10, so the temperature stays 1.0"]}}}})
check("upstream below-MIN_TYPE_N limitation -> INSUFFICIENT", low["types"]["choice"]["state"], evidence.INSUFFICIENT)

warned = calib({"training": {"laya_train_calibration": {"choice": {"items": 30, "temperature": 1.2,
        "issues": ["fitted on only 30 calibration items"]}}}})
check("upstream below-CALIB_WARN_N limitation -> INSUFFICIENT", warned["types"]["choice"]["state"], evidence.INSUFFICIENT)

clamped = calib({"training": {"laya_train_calibration": {"choice": {"items": 200, "temperature": 0.5,
        "issues": ["the fitted temperature 0.5 is on the [0.5, 3] clamp"]}}}})
check("clamped fit stays a visible limitation -> INSUFFICIENT", clamped["types"]["choice"]["state"], evidence.INSUFFICIENT)

clean = calib({"training": {"laya_train_calibration": {"choice": {"items": 400, "temperature": 1.1, "issues": []}}}})
check("clean supported fit -> PRESENT", clean["types"]["choice"]["state"], evidence.PRESENT)
check("all-clean overall -> PRESENT", clean["state"], evidence.PRESENT)

mixed = calib({"training": {"laya_train_calibration": {
    "choice": {"items": 400, "temperature": 1.1, "issues": []},
    "score": {"items": 0, "temperature": 1.0, "issues": []}}}})
check("mixed types overall -> INSUFFICIENT", mixed["state"], evidence.INSUFFICIENT)

malformed = calib({"training": {"laya_train_calibration": {"choice": {"temperature": 1.0}}}})
# `get`, not indexing: this is the entry shape whose whole result changed, and a suite that aborts on
# a missing key never reaches the checks that would say which one went.
check("malformed entry -> UNKNOWN", malformed.get("types", {}).get("choice", {}).get("state"),
      evidence.UNKNOWN)

# ------------------------------------------------------ what the overall may claim
# `INSUFFICIENT` is the state that reports a limitation against a fit that was made, so it is a
# claim about a fit. Folded blindly it also came out for a block that records no type at all, or one
# whose entries this module cannot read -- manufactured from absent input, which the module docstring
# rules out. `format_summary` prints a reason only for an UNKNOWN verdict, so the wrong state also
# hid the only explanation a reader gets. `MISSING` stays reserved for what the artifact positively
# records: zero items.
check("every entry unreadable -> UNKNOWN, not a claim about a fit",
      calib({"training": {"laya_train_calibration": {"choice": {"temperature": 1.0}}}})["state"],
      evidence.UNKNOWN)
check("an empty calibration block -> UNKNOWN",
      calib({"training": {"laya_train_calibration": {}}})["state"], evidence.UNKNOWN)
_allbad = calib({"training": {"laya_train_calibration": {"choice": "junk", "score": []}}})
# Both read through `get`, not indexing: a result that lost its reason or its per-type map has to
# fail these checks by name, not abort the suite before the rest of it runs.
check("and the two reasons say which shape was found",
      [_allbad.get("reason"), calib({"training": {"laya_train_calibration": {}}}).get("reason")],
      ["every calibration entry is malformed; nothing to read",
       "calibration block records no question type"])
check("the per-type map is still there to read beside that reason",
      {name: t["state"] for name, t in (_allbad.get("types") or {}).items()},
      {"choice": evidence.UNKNOWN, "score": evidence.UNKNOWN})
# Over-correcting is the failure mode to hold against: one entry this module can judge is enough to
# keep the verdict that reports a shortfall, and enough to keep `PRESENT` and `MISSING` folding.
check("an unreadable entry beside a clean one is still INSUFFICIENT",
      calib({"training": {"laya_train_calibration": {
          "choice": {"items": 400, "temperature": 1.1, "issues": []},
          "score": {"temperature": 1.0}}}})["state"], evidence.INSUFFICIENT)
check("an unreadable entry beside a zero-item one is still INSUFFICIENT",
      calib({"training": {"laya_train_calibration": {
          "choice": {"items": 0, "temperature": 1.0, "issues": []},
          "score": {"temperature": 1.0}}}})["state"], evidence.INSUFFICIENT)
check("every type at zero items -> MISSING",
      calib({"training": {"laya_train_calibration": {
          "choice": {"items": 0, "temperature": 1.0, "issues": []},
          "score": {"items": 0, "temperature": 1.0, "issues": []}}}})["state"], evidence.MISSING)

# ------------------------------------------------------ one numeric source of truth
# `items` is reported here, never judged: #964 deleted the local `_thresholds()` that compared it
# against copies of `MIN_TYPE_N`/`CALIB_WARN_N`, and the comment describing that mechanism stayed
# behind. These are the checks the deleted mechanism never had -- a re-introduced floor has to
# survive all of them, on both sides of both upstream floors.
for _n in (1, 5, 9, 10, 49, 50, 300, 5000):
    _t = calib({"training": {"laya_train_calibration": {
        "choice": {"items": _n, "temperature": 1.1, "issues": []}}}})
    check("items=%d with no recorded issue -> PRESENT" % _n, _t["types"]["choice"]["state"], evidence.PRESENT)

# The same counts with a limitation recorded, and the state flips -- so the loop above shows that
# `issues` decides, not that a large `items` does.
_states = {calib({"training": {"laya_train_calibration": {
    "choice": {"items": _n, "temperature": 1.1,
               "issues": ["fitted on only %d calibration items" % _n]}}}})["types"]["choice"]["state"]
    for _n in (5, 9, 10, 49, 50, 300)}
check("the limitation text alone decides, at every count", _states, {evidence.INSUFFICIENT})

# The deleted mechanism also exported where its numbers came from. Nothing in this module's result
# may claim a threshold provenance, because there is no threshold to have a provenance about.
check("no threshold provenance in the result",
      "threshold_source" in calib({"training": {"laya_train_calibration": {
          "choice": {"items": 400, "temperature": 1.1, "issues": []}}}}), False)

_src = inspect.getsource(evidence)
check_true("the module comments name the artifact as the authority", "second numeric source of truth" in _src)
check_true("no comment still promises the deleted fallback",
           "_FALLBACK" not in _src and "fall back to the documented values" not in _src)

# ------------------------------------------------------ report identity
full_report = {"config": {"schema": "laya-evals-report/1", "dataset_sha256": "a",
                          "questions_sha256": "b", "laya_version": "0.3.28"}, "overall": {}}
rep = evidence.inspect_report(full_report)
check("full identity -> PRESENT", rep["state"], evidence.PRESENT)

old_report = {"config": {"schema": "laya-evals-report/1"}, "overall": {}}
rep2 = evidence.inspect_report(old_report)
check("missing identity fields -> UNKNOWN", rep2["state"], evidence.UNKNOWN)

toplevel = {"schema": "laya-evals-report/1", "dataset_sha256": "a", "questions_sha256": "b",
            "config": {}}
check("identity at top level still read", evidence.inspect_report(toplevel)["state"], evidence.PRESENT)

# ------------------------------------------------------ identity vocabulary pinned to laya.evals
# `laya/evidence.py` copies the report-identity vocabulary instead of importing `laya.evals` (numpy at
# module scope), and a copy is only safe while it is held against the original -- the same way
# `tests/test_conformal_abstention.py` holds `confidence.BUCKET_QTYPES` against `common.temp_bucket`.
# Unpinned, a key renamed upstream would make every report read as identity-incomplete here, and a key
# added upstream would read as identity-complete; both silently, in the verdict this module exists to
# give. These four checks fail in both directions, on either side of the copy.
from laya import evals  # noqa: E402

check("the copied identity vocabulary is evals' own, in order",
      list(evidence._IDENTITY_KEYS), list(evals._IDENTITY_KEYS))
check("the provenance key is recorded upstream but not compared there",
      [k for k in evidence._PROVENANCE_KEYS if k not in evals._IDENTITY_KEYS],
      list(evidence._PROVENANCE_KEYS))
check("checkpoint <-> report conflicts are the identity vocabulary minus the report tag",
      sorted(set(evals._IDENTITY_KEYS) - set(evidence._IDENTITY_CONFLICT_KEYS)), ["schema"])
check("inspect_report reads exactly its declared vocabulary",
      sorted(evidence.inspect_report({"config": {}})["fields"]),
      sorted(evidence._IDENTITY_KEYS + evidence._PROVENANCE_KEYS))

# ... and `schema` stays out of the conflict set on purpose: two artifacts that agree on the
# deterministic digests but carry different report tags are still the same measurement.
check("a differing schema is not a checkpoint <-> report conflict",
      evidence.relate({"dataset_sha256": "a"},
                      {"config": {"dataset_sha256": "a", "schema": "laya-evals-report/2"}})["state"],
      evidence.UNKNOWN)

# `docs/evals.md` carries the same vocabulary a third time -- it teaches that the identity fields
# "are reported with the same semantics as `run`", and names the pair a conflict is claimed on -- so
# the page is pinned to the code as well, the way `test_docs_and_the_cli_name_the_same_flags` pins
# the flag list. A missing sentence reads as an empty list, which fails the check rather than
# crashing the suite.
import re  # noqa: E402

_DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "evals.md")
with open(_DOCS, encoding="utf-8") as _handle:
    _page = " ".join(_handle.read().split())
_named = _page.split("identity fields (", 1)[1].split(")", 1)[0] if "identity fields (" in _page else ""
check("the documented identity fields are the ones the code reads",
      sorted(re.findall(r"`([^`]+)`", _named)),
      sorted(evidence._IDENTITY_KEYS + evidence._PROVENANCE_KEYS))
_claimed = (_page.split("both artifacts expose the same ", 1)[1].split("with different values", 1)[0]
            if "both artifacts expose the same " in _page else "")
check("the documented conflict pair is the one the code compares",
      sorted(re.findall(r"`([^`]+)`", " ".join(_claimed.split("/")))),
      sorted(evidence._IDENTITY_CONFLICT_KEYS))

# The page now states the fold as well, including the case this suite caught the helper overstating.
# Read the verdict off the sentence rather than quoting it: a page that moves the claim elsewhere, or
# a code change that moves the state, fails here instead of quietly disagreeing.
_block = (_page.split("A block that records no question type", 1)[1].split("was found", 1)[0]
          if "A block that records no question type" in _page else "")
check("the page's verdict for a calibration block with nothing to read is the code's",
      re.findall(r"is `([A-Z]+)`", _block),
      [calib({"training": {"laya_train_calibration": {}}})["state"]])

# ------------------------------------------------------ relationship
config = {"training": {"laya_train": {"laya_version": "0.3.28"}}}
match = evidence.relate(config, {"config": {"laya_version": "0.3.28"}})
check("same laya_version alone must NOT prove PRESENT", match["state"], evidence.UNKNOWN)
conflict = evidence.relate(config, {"config": {"laya_version": "0.3.20"}})
check("different laya_version must NOT become INCOMPARABLE", conflict["state"], evidence.UNKNOWN)
conflict_ds = evidence.relate({"dataset_sha256": "a"}, {"config": {"dataset_sha256": "b"}})
check("deterministic identity conflict -> INCOMPARABLE", conflict_ds["state"], evidence.INCOMPARABLE)
unknown = evidence.relate(config, {"config": {}})
check("no shared identity field -> UNKNOWN", unknown["state"], evidence.UNKNOWN)

# ------------------------------------------------------ CLI on a realistic artifact pair
tmp = tempfile.mkdtemp()
ckpt = os.path.join(tmp, "ckpt")
os.makedirs(ckpt)
with open(os.path.join(ckpt, "rl_agent_config.json"), "w", encoding="utf-8") as f:
    json.dump({"training": {"laya_train": {"laya_version": "0.3.28"}, "laya_train_calibration": {
        "choice": {"items": 400, "temperature": 1.1, "issues": []},
        "score": {"items": 30, "temperature": 1.0, "issues": ["fitted on only 30 calibration items"]}}}}, f)
report_path = os.path.join(tmp, "report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump({"config": {"schema": "laya-evals-report/1", "dataset_sha256": "a",
                          "questions_sha256": "b", "laya_version": "0.3.28"}, "overall": {}}, f)

rc = evals_cli.main(["evidence", "--checkpoint", ckpt, "--report", report_path])
check("evidence CLI returns 0 on realistic pair", rc, 0)

rc_bad = evals_cli.main(["evidence", "--checkpoint", os.path.join(tmp, "nope")])
check("missing checkpoint dir -> usage error 2", rc_bad, 2)

# The summary is where a reader meets the calibration verdict, and it prints a reason only for an
# UNKNOWN state -- so a block that records nothing had to reach UNKNOWN for any explanation to
# appear at all. On INSUFFICIENT it printed a claim about a fit and nothing to check it against.
_empty_cal = os.path.join(tmp, "ckpt-empty-calibration")
os.makedirs(_empty_cal)
with open(os.path.join(_empty_cal, "rl_agent_config.json"), "w", encoding="utf-8") as f:
    json.dump({"training": {"laya_train": {"base": "x"}, "laya_train_calibration": {}}}, f)
_cal_summary = evidence.format_summary(evidence.inspect_checkpoint(_empty_cal))
check_true("the summary reads an empty calibration block as UNKNOWN",
           "calibration: UNKNOWN" in _cal_summary)
check_true("and prints why", "calibration block records no question type" in _cal_summary)

# ------------------------------------------------------ no torch required
import subprocess
_BLOCK = (
    "import sys, os, json, tempfile\n"
    "sys.path.insert(0, os.getcwd())\n"
    "class B:\n"
    "    def find_spec(self, name, path, target=None):\n"
    "        if name == 'torch' or name.startswith('torch.'):\n"
    "            raise ImportError('torch blocked')\n"
    "sys.meta_path.insert(0, B())\n"
    "tmp = tempfile.mkdtemp(); ckpt = os.path.join(tmp, 'c'); os.makedirs(ckpt)\n"
    "with open(os.path.join(ckpt, 'rl_agent_config.json'), 'w') as f:\n"
    "    json.dump({'training': {'laya_train_calibration': {'choice': {'items': 5, 'temperature': 1.0, 'issues': []}}}}, f)\n"
    "from laya import evals_cli, evidence\n"
    "rc = evals_cli.main(['evidence', '--checkpoint', ckpt])\n"
    "res = evidence.inspect_checkpoint(ckpt)\n"
    "print('RC', rc, 'TORCH', 'torch' in sys.modules)\n"
)
proc = subprocess.run([sys.executable, "-c", _BLOCK], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      capture_output=True, text=True, timeout=120)
print(proc.stdout.strip(), proc.stderr.strip()[-300:])
out = proc.stdout
check_true("evidence CLI works with torch blocked", "RC 0" in out)
check_true("torch not imported when blocked", "TORCH False" in out)

if FAIL:
    print("\nFAILURES:", FAIL)
    sys.exit(1)
print("\nall %d checks passed" % len(PASS))
