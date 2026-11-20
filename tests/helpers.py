import json

from auditgym.env import AuditEnv
from auditgym.generator import generate
from auditgym.policies import make_policy

TRUTH_TOKENS = ("expected_amount_minor", "evidence_sets", "issue_id", "ISS-", "provenance", "truth",
                '"variant"', '"scenario"', '"seed"')


def first_seed(config, pred, start=0, limit=200):
    for s in range(start, start + limit):
        pub, prv = generate(s, config)
        if pred(pub, prv):
            return s, pub, prv
    raise AssertionError("no seed satisfies predicate")


def erroneous(config="smoke"):
    return first_seed(config, lambda p, v: len(v["truth"]["issues"]) >= 3)


def clean(config="clean"):
    return first_seed(config, lambda p, v: v["truth"]["clean"])


def play(env, policy_name, seed=None, config=None, options=None):
    obs, info = env.reset(seed=seed, config=config, options=options)
    pol = make_policy(policy_name, 0)
    if pol.privileged:
        pol.set_truth(env.truth)
    pol.reset(obs, info)
    traj = [(None, obs, 0.0, False, False)]
    while True:
        a = pol.act(obs, info)
        obs, r, te, tr, info = env.step(a)
        traj.append((a, obs, r, te, tr))
        if te or tr:
            return traj, r, info


def oracle_submission(pub, prv):
    return {"episode_id": pub["manifest"]["episode_id"], "findings": [
        {"type": t["type"], "affected_ids": list(t["affected_ids"]), "claimed_amount_minor": t["expected_amount_minor"],
         "currency": t["currency"], "evidence_ids": list(t["evidence_sets"][0]), "explanation": "x"}
        for t in prv["truth"]["issues"]]}


def blob(obj):
    return json.dumps(obj, sort_keys=True)
