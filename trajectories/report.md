# Baseline diagnostics

Means over episodes. P and R are 0 on clean episodes by definition (zero denominator), so mean P/R of a perfect policy equals the share of non-clean episodes; F1 and return are the primary numbers.

| policy | episodes | P | R | F1 | return | return (clean eps) | return (eps with interacting errors) | steps | wall s | TP/FP/FN |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 10 | 0.000 | 0.000 | 0.000 | -1.000 | -1.000 | -1.000 | 19.0 | 0.015 | 0/9/47 |
| reconciler | 10 | 0.142 | 0.372 | 0.195 | -0.645 | -1.000 | -0.508 | 22.8 | 0.036 | 25/179/22 |
| flag_all | 10 | 0.003 | 0.020 | 0.006 | -0.993 | -1.000 | -0.990 | 19.3 | 0.030 | 1/299/46 |
| oracle | 10 | 0.800 | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 | 1.0 | 0.008 | 47/0/0 |
| expert | 10 | 0.800 | 0.800 | 1.000 | 0.944 | 0.953 | 0.941 | 28.9 | 0.029 | 47/0/0 |

## By episode group

| policy | return (train) | F1 (train) |
|---|---|---|
| random | -1.000 | 0.00 |
| reconciler | -0.646 | 0.20 |
| flag_all | -0.993 | 0.01 |
| oracle | 1.000 | 1.00 |
| expert | 0.944 | 1.00 |

## Measured failure analysis

Reasons come from the grader diagnostics (first failing criterion against the issue with the same affected ids; `no_such_issue` = no issue touches those ids).

| policy | FP by reason | FP by claimed type | FN by type/variant |
|---|---|---|---|
| random | no_such_issue x9 | payment_allocation x3, period_cutoff x3, fx_conversion x2, duplicate_invoice x1 | duplicate_invoice/duplicate_document x11, payment_allocation/missing x11, period_cutoff/tz_boundary x5, fx_conversion/utc_date x5, duplicate_invoice/double_posting x4, payment_allocation/partial x4 |
| reconciler | no_such_issue x157, wrong_type x7, wrong_affected_ids x7, amount_out_of_tolerance x6, insufficient_evidence x2 | payment_allocation x89, fx_conversion x47, period_cutoff x22, duplicate_invoice x21 | payment_allocation/missing x6, period_cutoff/tz_boundary x5, fx_conversion/utc_date x5, payment_allocation/partial x4, duplicate_invoice/duplicate_document x1, duplicate_invoice/double_posting x1 |
| flag_all | no_such_issue x273, wrong_affected_ids x20, wrong_type x6 | payment_allocation x80, duplicate_invoice x79, fx_conversion x70, period_cutoff x70 | duplicate_invoice/duplicate_document x11, payment_allocation/missing x11, period_cutoff/tz_boundary x5, fx_conversion/utc_date x5, payment_allocation/partial x4, period_cutoff/backdated x3 |
| oracle | - | - | - |
| expert | - | - | - |
