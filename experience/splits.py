"""Pinned IDs and conservative goal-group partitioning; never read test outcomes."""
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

TRAIN_SHA = "9b7ce7f214c738de96d0048919e08e8241c5c2bca48843c53dae5098b938e28d"
TEST_SHA = "78fbe5ed7ba545f1a2ee08397a5e740f622872b6a3ee70315d349c4287675c79"
CATALOG_SHA = "f16f8cc9c57a8066e45e1be2c61201dd8a10ee90c6fc6164fcdb22375ac04e94"
REVISION = "99d0b7923bf126d9c6cdea5f362d2d41ccb5d493"
RESERVED = {"sciworld_349", "sciworld_1709", "sciworld_2997", "sciworld_4509"}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def goal_key(goal):
    # Verified old AgentTraj/current SDK wording difference in thermometer goals.
    canonical = re.sub(r"\s+", " ", goal).strip().casefold()
    canonical = canonical.replace(" degrees celsius", " degrees")
    return digest(canonical)


def prepare(train_ids, test_ids, catalog, goal_metadata, test_metadata, native_goal_metadata=None):
    for path, expected in [(train_ids, TRAIN_SHA), (test_ids, TEST_SHA), (catalog, CATALOG_SHA)]:
        if file_hash(path) != expected:
            raise ValueError("Unpinned split/catalog source: " + str(path))
    spec = importlib.util.spec_from_file_location("phase3_catalog", catalog)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train = [r["item_id"] for r in json.loads(Path(train_ids).read_text())]
    tests = [r["item_id"] for r in json.loads(Path(test_ids).read_text())]
    if len(set(train)) != len(train) or len(set(tests)) != len(tests):
        raise ValueError("Duplicate source IDs")
    # Read ONLY identity/goal fingerprints from the evaluation metadata.
    metadata = json.loads(Path(test_metadata).read_text())["tasks"]
    test_hashes = {r["task_id"]: r["task_description_sha256"] for r in metadata}
    if set(test_hashes) != set(tests):
        raise ValueError("Test identity metadata mismatch")
    native_tests = {}
    if native_goal_metadata:
        native = json.loads(Path(native_goal_metadata).read_text())["records"]
        native_tests = {r["task_id"]: r for r in native if r["task_id"] in test_hashes}
        if set(native_tests) != set(tests):
            raise ValueError("Incomplete native test goal identity audit")
        for item, r in native_tests.items():
            m = module.resolve_item(item)
            if (r["task_name"], r["variation_id"]) != (m["task_name"], m["variation_id"]):
                raise ValueError("Native canonical identity mismatch")
            if digest(r["goal"]) != test_hashes[item]:
                raise ValueError("Native goal differs from existing audited test identity")
    goals = {}
    for row in json.loads(Path(goal_metadata).read_text()):
        text = row["conversations"][2]["value"]
        marker = "\nThis room is called "
        if marker in text:
            goals[row["item_id"]] = text.split(marker, 1)[0]
    overlap = set(train) & set(tests)
    verified = [k for k in overlap if k in goals]
    if any(digest(goals[k]) != test_hashes[k] for k in verified):
        raise ValueError("Goal extraction disagrees with audited native hashes")
    reserved_groups = {goal_key(goals[k]) for k in RESERVED if k in goals}
    raw_test_hashes = set(test_hashes.values())
    test_normal_groups = {goal_key(goals[k]) for k in tests if k in goals}
    test_normal_groups.update(goal_key(r["goal"]) for r in native_tests.values())
    groups, excluded = {}, {}
    all_canonical = []
    for item in sorted(set(train) | set(tests)):
        m = module.resolve_item(item)
        all_canonical.append((m["task_name"], m["variation_id"]))
    if len(set(all_canonical)) != len(all_canonical):
        raise ValueError("Aliased task/variation IDs")
    for item in train:
        reason = None
        if item in tests:
            reason = "test_id"
        elif item not in goals:
            reason = "unverified_goal_format"
        elif digest(goals[item]) in raw_test_hashes or goal_key(goals[item]) in test_normal_groups:
            reason = "test_goal_duplicate"
        elif item == "sciworld_4509" or goal_key(goals[item]) == goal_key(goals.get("sciworld_4509", "")):
            reason = "demo_goal_reserved"
        if reason:
            excluded[item] = reason
            continue
        m = module.resolve_item(item)
        row = {"task_id": item, "index": int(item.rsplit("_", 1)[1]),
               "taskName": m["task_name"], "variationIdx": m["variation_id"],
               "category": m["category"], "goal_sha256": digest(goals[item]),
               "goal_group": goal_key(goals[item])}
        groups.setdefault(row["goal_group"], []).append(row)
    dev_groups = {g for g in groups if g in reserved_groups}
    # Whole goal groups move together. Quotas are target instance counts, not an
    # assertion that rounding and grouping preserve exactly 10%.
    quotas = {}
    categories = sorted({r["category"] for rows in groups.values() for r in rows} &
                        {"measure", "test_conductivity", "find", "chemistry_mix", "lifespan"})
    for category in categories:
        candidates = [r for rows in groups.values() for r in rows if r["category"] == category]
        quota = max(1, len(candidates) // 10)
        quotas[category] = quota
        selected = sum(r["category"] == category for g in dev_groups for r in groups[g])
        ranked = sorted(candidates, key=lambda r: (digest("agentwmv3-dev-v1|" + r["task_id"]), r["index"]))
        for row in ranked:
            if selected >= quota:
                break
            group = row["goal_group"]
            if group not in dev_groups:
                dev_groups.add(group)
                selected += sum(r["category"] == category for r in groups[group])
    partitions = {"train": [], "development": [], "test": []}
    for group, rows in sorted(groups.items()):
        split = "development" if group in dev_groups else "train"
        partitions[split].extend(dict(r, split=split) for r in rows)
    for item in tests:
        m = module.resolve_item(item)
        partitions["test"].append({"task_id": item, "taskName": m["task_name"],
            "variationIdx": m["variation_id"], "goal_sha256": test_hashes[item], "split": "test",
            "goal_group": goal_key(native_tests[item]["goal"]) if item in native_tests else None})
    manifest = {"schema": "scienceworld-experience-split-v1", "source_revision": REVISION,
        "source_hashes": {"train_ids": file_hash(train_ids), "test_ids": file_hash(test_ids),
                          "catalog": file_hash(catalog), "goal_metadata": file_hash(goal_metadata),
                          "test_metadata": file_hash(test_metadata)},
        "native_goal_metadata_sha256": file_hash(native_goal_metadata) if native_goal_metadata else None,
        "algorithm": "goal-group SHA256 canonical-ID ranking; category instance quotas; reserved prior diagnostic goal groups",
        "partitions": partitions, "excluded": excluded,
        "audit": {"source_train": len(train), "source_test": len(tests), "id_overlap": len(overlap),
            "parsed_goal_count": len(goals), "verified_overlap_hashes": len(verified),
            "excluded_by_reason": dict(Counter(excluded.values())), "quotas": quotas,
            "counts": {k: len(v) for k, v in partitions.items()},
            "goal_groups": {s: len({r["goal_group"] for r in partitions[s]}) for s in ("train", "development")},
            "template_overlap_expected": True,
            "native_test_goal_count": len(native_tests),
            "goal_normalization": "casefold/whitespace/degrees-celsius alias v2",
            "limitations": ["Exact normalized goal duplicates excluded; semantic near-duplicates not fully ruled out",
                "Different variations/goals can share task templates and physical layouts",
                "Legacy trajectories used for initial goal identity only, never case evidence; ID-less expert files rejected"]}}
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest, require_native=False):
    parts = manifest["partitions"]
    ids = {s: {r["task_id"] for r in rows} for s, rows in parts.items()}
    if any(ids[a] & ids[b] for a, b in [("train", "development"), ("train", "test"), ("development", "test")]):
        raise ValueError("Split task ID leakage")
    a = {r["goal_group"] for r in parts["train"]}
    b = {r["goal_group"] for r in parts["development"]}
    test_hashes = {r["goal_sha256"] for r in parts["test"]}
    test_groups = {r.get("goal_group") for r in parts["test"]} - {None}
    if a & b or (a | b) & test_groups or any(r["goal_sha256"] in test_hashes for s in ("train", "development") for r in parts[s]):
        raise ValueError("Split goal leakage")
    if ids["train"] & RESERVED:
        raise ValueError("Previously used development/demo instance in memory pool")
    if require_native and any(r.get("goal_group") is None for r in parts["test"]):
        raise ValueError("Memory requires final native-normalized test goal identities, not bootstrap split")


def select_diagnostic(manifest, split, per_family=2):
    result = []
    for family in ("test-conductivity", "use-thermometer", "find-living-thing"):
        candidates = [r for r in manifest["partitions"][split] if r["taskName"] == family]
        candidates.sort(key=lambda r: (digest("agentwmv3-phase3-collection-v1|" + r["task_id"]), r["index"]))
        seen = set()
        for row in candidates:
            if row["goal_group"] in seen:
                continue
            result.append(row)
            seen.add(row["goal_group"])
            if len(seen) == per_family:
                break
    return result
