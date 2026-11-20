"""Policy adapter interface and shared helpers.

A policy only sees observations and returns JSON actions:
    policy.reset(observation, info)
    action = policy.act(observation, info)
"""

from __future__ import annotations

from decimal import Decimal

from ..constants import PAGE_LIMIT


class Policy:
    name = "base"
    privileged = False   # True only for the evaluator-side oracle

    def reset(self, observation, info):
        self.manifest = observation["manifest"]
        self.policy_doc = observation["policy"]
        self.episode_id = observation["episode_id"]
        self.steps_remaining = observation["steps_remaining"]

    def act(self, observation, info):  # pragma: no cover - interface
        raise NotImplementedError

    def submit(self, findings):
        return {"action": "submit", "findings": {"episode_id": self.episode_id, "findings": findings}}


class TableReader:
    """Pages through tables with list_records and accumulates rows."""

    def __init__(self, tables):
        self.queue = list(tables)
        self.rows = {t: [] for t in tables}
        self.cursor = None
        self.pending = None

    def done(self):
        return not self.queue

    def next_action(self):
        t = self.queue[0]
        self.pending = t
        return {"action": "list_records", "table": t, "cursor": self.cursor, "limit": PAGE_LIMIT}

    def consume(self, observation):
        if self.pending is None:
            return
        res = observation.get("result") or {}
        if observation.get("error") or res.get("table") != self.pending:
            # unexpected: skip this table rather than loop forever
            self.queue.pop(0)
            self.cursor = None
        else:
            self.rows[self.pending].extend(res["rows"])
            self.cursor = res["next_cursor"]
            if self.cursor is None:
                self.queue.pop(0)
        self.pending = None

    def pages_needed(self, manifest):
        counts = manifest["counts"]
        return sum(-(-counts[t] // PAGE_LIMIT) for t in self.queue)


def base_currency_from(manifest):
    return manifest["base_currency"]


def as_decimal(x):
    return Decimal(str(x))
