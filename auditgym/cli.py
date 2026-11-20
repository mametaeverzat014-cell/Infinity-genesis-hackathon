"""Command line: python -m auditgym <command>

  check         the single official check (tests + data verification + baselines + invariance)
  generate      generate one episode directory from (seed, config)
  build-splits  (re)generate the open train/dev episodes listed in episodes/splits.json
  verify        verify episode directories or fresh seeds
  run           run policies, write trajectories / grader log / summary
  demo          verbose run of one policy on one episode
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import unittest

from . import __version__
from .config import load_config
from .constants import ENV_VERSION
from .env import AuditEnv
from .episode import public_file_texts, read_private, read_public, write_episode
from .generator import generate
from .policies import BASELINES, make_policy
from .runner import run_policy, summarize
from .verify import verify_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPISODES = os.path.join(ROOT, "episodes")

# Seeds used by the official check. Closed evaluation must use different, secret, high-entropy seeds.
CHECK_SUITE = ([{"seed": 910000 + i, "config": "eval_mixed", "tag": "mixed"} for i in range(10)]
               + [{"seed": 920000 + i, "config": "eval_clean", "tag": "clean"} for i in range(5)]
               + [{"seed": 930000 + i, "config": "eval_interaction", "tag": "interaction"} for i in range(5)])
PAIR_TRANSFORM = {"id_salt": "paired-check", "order_salt": "paired-check", "shuffle_records": True}


def _p(*a):
    print(*a, flush=True)


def load_splits():
    with open(os.path.join(EPISODES, "splits.json"), encoding="utf-8") as fh:
        return json.load(fh)


def split_dirs(split):
    d = os.path.join(EPISODES, split)
    if not os.path.isdir(d):
        return []
    return [os.path.join(d, x) for x in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, x))]


# ------------------------------------------------------------------ commands
def cmd_generate(a):
    pub, prv = generate(a.seed, a.config)
    write_episode(a.out, pub, None if a.public_only else prv)
    _p("wrote", a.out, pub["manifest"]["episode_id"], "issues:", len(prv["truth"]["issues"]))


def cmd_build_splits(a):
    splits = load_splits()
    index = {}
    for split in ("train", "dev"):
        d = os.path.join(EPISODES, split)
        if os.path.isdir(d):
            shutil.rmtree(d)
        index[split] = []
        for i, spec in enumerate(splits[split]):
            pub, prv = generate(spec["seed"], spec["config"], secret="")  # open episodes: no secret
            eid = pub["manifest"]["episode_id"]
            out = os.path.join(d, "%02d_%s" % (i, eid))
            write_episode(out, pub, prv)
            index[split].append({"dir": os.path.relpath(out, EPISODES), "episode_id": eid, "seed": spec["seed"],
                                 "config": spec["config"], "issues": len(prv["truth"]["issues"]),
                                 "clean": prv["truth"]["clean"],
                                 "interaction": any(t["scenario"] for t in prv["truth"]["issues"]),
                                 "counts": pub["manifest"]["counts"]})
            _p(split, out, len(prv["truth"]["issues"]), "issues")
    with open(os.path.join(EPISODES, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
        fh.write("\n")


def verify_dirs(dirs, regenerate=True):
    out = []
    for d in dirs:
        pub, prv = read_public(d), read_private(d)
        rep = verify_episode(pub, prv)
        if regenerate:
            prov = prv["provenance"]
            seed = prov["seed"]
            seed = int(seed) if seed.lstrip("-").isdigit() else seed
            pub2, _ = generate(seed, prov["config"], secret=None if prov.get("seed_secret_used") else "")
            texts = public_file_texts(pub2)
            same = all(open(os.path.join(d, "public", n), encoding="utf-8").read() == t for n, t in texts.items())
            rep["checks"]["regenerates_byte_identical"] = same
            rep["ok"] = rep["ok"] and same
        rep["dir"] = os.path.relpath(d, ROOT)
        out.append(rep)
    return out


def cmd_verify(a):
    reps = []
    if a.seeds:
        lo, hi = [int(x) for x in a.seeds.split("-")]
        for s in range(lo, hi + 1):
            pub, prv = generate(s, a.config)
            reps.append(verify_episode(pub, prv))
    else:
        dirs = a.dirs or (split_dirs("train") + split_dirs("dev"))
        reps = verify_dirs(dirs)
    bad = [r for r in reps if not r["ok"]]
    for r in bad:
        _p("FAIL", r["episode_id"], [k for k, v in r["checks"].items() if not v], r["errors"][:3])
    _p("verified %d episodes, %d failed" % (len(reps), len(bad)))
    return 1 if bad else 0


def _episode_specs(a):
    if a.split:
        return [{"episode_dir": d, "tag": a.split} for d in split_dirs(a.split)]
    lo, hi = [int(x) for x in a.seeds.split("-")]
    return [{"seed": s, "config": a.config} for s in range(lo, hi + 1)]


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    specs = _episode_specs(a)
    grader = os.path.join(a.out, "grader_log.jsonl")
    if os.path.exists(grader):
        os.remove(grader)
    results = []
    with open(os.path.join(a.out, "trajectories.jsonl"), "w", encoding="utf-8") as traj:
        for name in a.policies.split(","):
            results += run_policy(name, specs, mode=a.mode, isolation=a.isolation, trajectories=traj,
                                  grader_log=grader, policy_seed=a.policy_seed)
    with open(os.path.join(a.out, "results.jsonl"), "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    summ = summarize(results)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=2)
    from .report import markdown
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(markdown(results))
    _p(format_summary(summ))


def format_summary(summ):
    lines = ["| policy | episodes | P | R | F1 | return | return (clean eps) | return (eps with interacting errors) | steps | wall s | TP/FP/FN |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in summ.items():
        lines.append("| %s | %d | %.3f | %.3f | %.3f | %.3f | %s | %s | %.1f | %.3f | %d/%d/%d |" % (
            n, s["episodes"], s["precision"], s["recall"], s["f1"], s["return"],
            "-" if s["return_clean"] is None else "%.3f" % s["return_clean"],
            "-" if s["return_interaction"] is None else "%.3f" % s["return_interaction"],
            s["steps"], s["wall_s"], s["tp"], s["fp"], s["fn"]))
    return "\n".join(lines)


def cmd_demo(a):
    env = AuditEnv(mode="train")
    if a.episode_dir:
        obs, info = env.reset(options={"episode_dir": a.episode_dir})
    else:
        obs, info = env.reset(seed=a.seed, config=a.config)
    pol = make_policy(a.policy, 0)
    if pol.privileged:
        pol.set_truth(env.truth)
    pol.reset(obs, info)
    m = obs["manifest"]
    _p("episode %s  %s  base=%s  tz=%s  period=%s..%s" % (m["episode_id"], m["company_name"], m["base_currency"],
                                                          m["timezone"], m["period_start"], m["period_end"]))
    _p("records:", m["counts"])
    while True:
        act = pol.act(obs, info)
        obs, r, te, tr, info = env.step(act)
        desc = act.get("action")
        if desc == "list_records":
            res = obs.get("result") or {}
            if obs["kind"] == "terminal":
                res = res.get("last_result") or {}
            desc += " %s cursor=%s -> %d rows" % (act.get("table"), act.get("cursor"), len(res.get("rows") or []))
        elif desc == "get_document":
            desc += " " + act["doc_id"]
        elif desc == "submit":
            desc += " %d findings" % len(act["findings"]["findings"])
            for f in act["findings"]["findings"]:
                _p("    finding: %-19s %-40s %12d %s ev=%s" % (f["type"], ",".join(f["affected_ids"]),
                                                               f["claimed_amount_minor"], f["currency"],
                                                               len(f["evidence_ids"])))
        _p("step %3d  %-60s remaining=%d" % (obs["step"], desc, obs["steps_remaining"]))
        if te or tr:
            break
    _p("terminated=%s truncated=%s reward=%.4f metrics=%s" % (te, tr, r, info.get("metrics")))
    ev = env.evaluation()
    _p("grader diagnostics (evaluator side):")
    for d in ev["grade"]["findings"]:
        _p("   ", d)
    for d in ev["grade"]["missed"]:
        _p("   ", d)


# --------------------------------------------------------------------- check
def run_unit_tests():
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(ROOT, "tests"), top_level_dir=ROOT)
    res = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
    return {"run": res.testsRun, "failures": len(res.failures), "errors": len(res.errors),
            "ok": res.wasSuccessful()}


def check_invariance(n=20):
    """Paired checks: same seed with IDs changed and records shuffled -> same oracle answer and reward,
    and repeating a run with the same seed/actions reproduces the trajectory exactly."""
    out = []
    for i in range(n):
        spec = CHECK_SUITE[i % len(CHECK_SUITE)]
        cfg = load_config(spec["config"])
        cfg_t = load_config(dict(cfg, **PAIR_TRANSFORM))
        pa, va = generate(spec["seed"], cfg)
        pb, vb = generate(spec["seed"], cfg_t)
        key = lambda t: sorted((x["type"], x["variant"], x["expected_amount_minor"], len(x["affected_ids"]),
                                len(x["evidence_sets"])) for x in t["issues"])
        same_truth = key(va["truth"]) == key(vb["truth"])
        ids_changed = (set(r["row_id"] for r in pa["ledger"]) != set(r["row_id"] for r in pb["ledger"])
                       and [r["account"] for r in pa["ledger"]] != [r["account"] for r in pb["ledger"]])
        ra = run_policy("oracle", [{"seed": spec["seed"], "config": cfg}], mode="eval", isolation="inprocess")[0]
        rb = run_policy("oracle", [{"seed": spec["seed"], "config": cfg_t}], mode="eval", isolation="inprocess")[0]
        ea = run_policy("expert", [{"seed": spec["seed"], "config": cfg}], mode="eval", isolation="inprocess")[0]
        eb = run_policy("expert", [{"seed": spec["seed"], "config": cfg_t}], mode="eval", isolation="inprocess")[0]
        # replay: the same seed and the same actions give the same observations and reward
        import io
        b1, b2 = io.StringIO(), io.StringIO()
        r1 = run_policy("expert", [{"seed": spec["seed"], "config": cfg}], mode="train", isolation="inprocess", trajectories=b1)[0]
        r2 = run_policy("expert", [{"seed": spec["seed"], "config": cfg}], mode="train", isolation="inprocess", trajectories=b2)[0]
        replay = b1.getvalue() == b2.getvalue() and r1["return"] == r2["return"]
        ok = (same_truth and ids_changed and ra["return"] == rb["return"] == 1.0 and ra["f1"] == rb["f1"] == 1.0
              and ea["f1"] == eb["f1"] and replay)
        out.append({"seed": spec["seed"], "config": spec["config"], "ok": ok, "same_truth": same_truth,
                    "ids_and_order_changed": ids_changed, "oracle_return": [ra["return"], rb["return"]],
                    "expert_f1": [ea["f1"], eb["f1"]], "replay_identical": replay})
    return out


def cmd_check(a):
    t0 = time.time()
    report = {"env_version": ENV_VERSION, "package_version": __version__, "python": sys.version.split()[0],
              "platform": platform.platform(), "cpu_count": os.cpu_count()}
    ok = True
    out_dir = a.out
    os.makedirs(out_dir, exist_ok=True)

    _p("== 1/5 unit tests")
    if a.skip_tests:
        report["unit_tests"] = {"skipped": True}
    else:
        report["unit_tests"] = run_unit_tests()
        ok &= report["unit_tests"]["ok"]

    _p("== 2/5 open episodes: verification + byte-identical regeneration")
    reps = verify_dirs(split_dirs("train") + split_dirs("dev"))
    report["open_episodes"] = {"n": len(reps), "failed": [r["dir"] for r in reps if not r["ok"]]}
    ok &= not report["open_episodes"]["failed"] and len(reps) == 20
    _p("   %d episodes, %d failed" % (len(reps), len(report["open_episodes"]["failed"])))

    _p("== 3/5 fresh episodes: data and truth verification")
    fresh = []
    for spec in CHECK_SUITE + [{"seed": 940000 + i, "config": "stress"} for i in range(a.fresh)]:
        pub, prv = generate(spec["seed"], spec["config"])
        r = verify_episode(pub, prv)
        fresh.append({"seed": spec["seed"], "config": spec["config"], "ok": r["ok"], "checks": r["checks"],
                      "errors": r["errors"][:5], "issues": r["n_issues"]})
    report["fresh_episodes"] = {"n": len(fresh), "failed": [f for f in fresh if not f["ok"]]}
    ok &= not report["fresh_episodes"]["failed"]
    _p("   %d episodes, %d failed" % (len(fresh), len(report["fresh_episodes"]["failed"])))

    _p("== 4/5 baseline policies on the 20-episode check suite (eval mode, agents in separate processes)")
    grader = os.path.join(out_dir, "grader_log.jsonl")
    if os.path.exists(grader):
        os.remove(grader)
    results = []
    with open(os.path.join(out_dir, "trajectories.jsonl"), "w", encoding="utf-8") as traj:
        for name in list(BASELINES) + ["expert"]:
            results += run_policy(name, CHECK_SUITE, mode="eval", isolation="process", trajectories=traj,
                                  grader_log=grader)
    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    summ = summarize(results)
    report["baselines"] = summ
    from .report import markdown
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(markdown(results, "Check suite: baseline diagnostics (20 episodes, eval mode)"))
    sanity = {
        "oracle_return_is_max": summ["oracle"]["return"] == 1.0,
        "flag_all_is_bad": summ["flag_all"]["return"] <= -0.5,
        "random_below_oracle": summ["random"]["return"] < summ["oracle"]["return"],
        "expert_solves_without_truth": summ["expert"]["f1"] == 1.0,
        "no_leak_in_trajectories": _scan_trajectories(os.path.join(out_dir, "trajectories.jsonl")),
    }
    report["baseline_sanity"] = sanity
    ok &= all(sanity.values())
    _p(format_summary(summ))
    _p("   sanity:", sanity)

    _p("== 5/5 paired invariance and replay checks")
    inv = check_invariance(20)
    report["paired_checks"] = {"n": len(inv), "passed": sum(x["ok"] for x in inv), "details": inv}
    ok &= all(x["ok"] for x in inv)
    _p("   %d/%d passed" % (report["paired_checks"]["passed"], len(inv)))

    report["ok"] = bool(ok)
    report["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "check_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _p("\nCHECK %s in %.1fs  (report: %s)" % ("PASSED" if ok else "FAILED", report["elapsed_s"],
                                             os.path.relpath(os.path.join(out_dir, "check_report.json"), ROOT)))
    return 0 if ok else 1


def _scan_trajectories(path):
    """No truth-only field names or issue ids in anything the agent saw."""
    bad_tokens = ("expected_amount_minor", "evidence_sets", "issue_id", "ISS-", "provenance", "truth",
                  "\"variant\"", "\"scenario\"", "\"seed\"")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            blob = json.dumps({"observation": rec["observation"]})
            if any(t in blob for t in bad_tokens):
                return False
            if rec["mode"] == "eval" and rec["reward"] != 0.0:
                return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m auditgym")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--config", default="train")
    g.add_argument("--out", required=True)
    g.add_argument("--public-only", action="store_true")
    sub.add_parser("build-splits")
    v = sub.add_parser("verify")
    v.add_argument("dirs", nargs="*")
    v.add_argument("--seeds")
    v.add_argument("--config", default="train")
    r = sub.add_parser("run")
    r.add_argument("--policies", default=",".join(BASELINES))
    r.add_argument("--split", choices=["train", "dev"])
    r.add_argument("--seeds", default="0-9")
    r.add_argument("--config", default="train")
    r.add_argument("--mode", choices=["train", "eval"], default="eval")
    r.add_argument("--isolation", choices=["process", "inprocess"], default="process")
    r.add_argument("--policy-seed", type=int, default=0)
    r.add_argument("--out", default="runs/latest")
    d = sub.add_parser("demo")
    d.add_argument("--policy", default="expert")
    d.add_argument("--seed", type=int, default=1001)
    d.add_argument("--config", default="interaction")
    d.add_argument("--episode-dir")
    c = sub.add_parser("check")
    c.add_argument("--out", default="runs/check")
    c.add_argument("--fresh", type=int, default=20, help="extra stress-config episodes to verify")
    c.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args(argv)
    fn = {"generate": cmd_generate, "build-splits": cmd_build_splits, "verify": cmd_verify, "run": cmd_run,
          "demo": cmd_demo, "check": cmd_check}[a.cmd]
    rc = fn(a)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
