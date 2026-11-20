"""Markdown report with diagnostics and a measured failure analysis of each policy."""

from __future__ import annotations

from collections import Counter

from .runner import summarize


def _fmt_counter(c, n=6):
    if not c:
        return "-"
    return ", ".join("%s x%d" % (k, v) for k, v in c.most_common(n))


def failure_analysis(results):
    lines = ["| policy | FP by reason | FP by claimed type | FN by type/variant |", "|---|---|---|---|"]
    for name in dict.fromkeys(r["policy"] for r in results):
        rs = [r for r in results if r["policy"] == name]
        reasons = Counter(x for r in rs for x in r["fp_reasons"])
        fpt = Counter(x for r in rs for x in r["fp_types"])
        fnv = Counter("%s/%s" % (t, v) for r in rs for t, v in zip(r["fn_types"], r["fn_variants"]))
        lines.append("| %s | %s | %s | %s |" % (name, _fmt_counter(reasons), _fmt_counter(fpt), _fmt_counter(fnv)))
    return "\n".join(lines)


def per_tag(results):
    tags = sorted({r.get("tag") or "-" for r in results})
    head = "| policy | " + " | ".join("return (%s)" % t for t in tags) + " | F1 (" + ", ".join(tags) + ") |"
    lines = [head, "|---" * (len(tags) + 2) + "|"]
    for name in dict.fromkeys(r["policy"] for r in results):
        cells, f1s = [], []
        for t in tags:
            rs = [r for r in results if r["policy"] == name and (r.get("tag") or "-") == t]
            cells.append("%.3f" % (sum(r["return"] for r in rs) / len(rs)) if rs else "-")
            f1s.append("%.2f" % (sum(r["f1"] for r in rs) / len(rs)) if rs else "-")
        lines.append("| %s | %s | %s |" % (name, " | ".join(cells), ", ".join(f1s)))
    return "\n".join(lines)


def markdown(results, title="Baseline diagnostics"):
    from .cli import format_summary
    out = ["# " + title, "",
           "Means over episodes. P and R are 0 on clean episodes by definition (zero denominator), so mean P/R "
           "of a perfect policy equals the share of non-clean episodes; F1 and return are the primary numbers.", "",
           format_summary(summarize(results)), "", "## By episode group", "", per_tag(results), "",
           "## Measured failure analysis", "",
           "Reasons come from the grader diagnostics (first failing criterion against the issue with the same "
           "affected ids; `no_such_issue` = no issue touches those ids).", "", failure_analysis(results), ""]
    return "\n".join(out)
