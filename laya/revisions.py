"""Supply-chain integrity for published checkpoints: pinned revisions and digest checks.

Runtime loaders default to the mutable ``main`` revision of a Hub repo. For the published
checkpoints that is a silent-behavior-change risk: a compromised or accidentally changed
model repository would change routing, confidence, guardrail, and ONNX behavior with no
code change on the user's side. Loading a published repo without an explicit revision
therefore pins to the reviewed commit SHA below, and any loader accepts an optional
SHA-256 map to verify artifact integrity before weights reach the runtime.
"""
import hashlib
import os
from typing import Dict, Optional

# Reviewed commit SHAs of the published checkpoints. Bump a pin only after the new
# revision has been reviewed; treat an unexpected upstream SHA change as a prompt to
# review, not to blindly re-pin.
PINNED_REVISIONS: Dict[str, str] = {
    "convaiinnovations/laya": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
    "convaiinnovations/laya-multilingual": "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
    "convaiinnovations/laya-typed-decisions": "1a793eb568e6718f15941d08f85432581df534e3",
}


def resolve_revision(model_id_or_path: str, revision: Optional[str] = None) -> Optional[str]:
    """Pick the revision to download.

    An explicit `revision` always wins; a published repo falls back to its reviewed pin;
    anything else returns None so the Hub default (`main`) applies unchanged.
    """
    if revision:
        return revision
    return PINNED_REVISIONS.get(model_id_or_path)


def snapshot_revision(path: str) -> Optional[str]:
    """Commit SHA a Hub snapshot directory points at, or None for a plain directory.

    `snapshot_download` returns ``<cache>/snapshots/<sha>``; resolving symlinks keeps this
    correct when the snapshot entry is a link into the blob store.
    """
    real = os.path.realpath(path).rstrip(os.sep)
    parent, base = os.path.split(real)
    if os.path.basename(parent) == "snapshots" and base:
        return base
    return None


def verify_digests(model_dir: str, expected: Dict[str, str]) -> None:
    """Verify SHA-256 digests of files under `model_dir` against {relpath: hexdigest}.

    Raises FileNotFoundError when a listed file is absent and ValueError on a digest
    mismatch or an unsafe (absolute or escaping) relative path. Verification runs before
    any weight is parsed or executed, so a tampered artifact never reaches the runtime.
    """
    for rel, want in expected.items():
        rel_norm = str(rel).replace("\\", "/").lstrip("/")
        if not rel_norm or rel_norm == ".." or rel_norm.startswith("../") or "/../" in rel_norm:
            raise ValueError("laya: unsafe path in expected digests: %r" % (rel,))
        path = os.path.join(model_dir, rel_norm)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "laya: cannot verify %r: no such file under %s" % (rel, model_dir))
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        got = digest.hexdigest()
        if got.lower() != str(want).strip().lower():
            raise ValueError(
                "laya: SHA-256 mismatch for %s: expected %s, got %s. The artifact does "
                "not match the reviewed digest; refusing to load it." % (rel, want, got))
