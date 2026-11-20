"""Scripted expert auditor: reads every table through the tools, fetches the source documents
needed for duplicate checks, then applies the published policy (``solver.audit``).
Uses only what the observations contain - no truth, no seed."""

from __future__ import annotations

from ..solver import PublicBooks, audit, duplicate_candidate_groups
from .base import Policy, TableReader


class ExpertPolicy(Policy):
    name = "expert"

    def reset(self, observation, info):
        super().reset(observation, info)
        self.reader = TableReader(["ledger", "invoices", "payments", "fx"])
        self.doc_queue = None
        self.documents = {}
        self.pending_doc = None

    def _books(self):
        r = self.reader.rows
        return PublicBooks(self.manifest, r["ledger"], r["invoices"], r["payments"], r["fx"])

    def act(self, observation, info):
        remaining = observation["steps_remaining"]
        if observation.get("kind") != "reset":
            if self.pending_doc is not None:
                res = observation.get("result") or {}
                if res.get("document"):
                    self.documents[self.pending_doc] = res["document"]
                self.pending_doc = None
            else:
                self.reader.consume(observation)
        if remaining <= 1:
            return self._final()
        if not self.reader.done():
            return self.reader.next_action()
        if self.doc_queue is None:
            books = self._books()
            ids = []
            for group in duplicate_candidate_groups(books):
                ids.extend(books.invoices[i]["doc_id"] for i in group)
            self.doc_queue = ids
        if self.doc_queue:
            self.pending_doc = self.doc_queue.pop(0)
            return {"action": "get_document", "doc_id": self.pending_doc}
        return self._final()

    def _final(self):
        if not self.reader.done():
            return self.submit([])  # partial reads would turn missing rows into false findings
        try:
            findings = audit(self._books(), self.documents)
        except Exception:
            findings = []
        return self.submit(findings[:30])
