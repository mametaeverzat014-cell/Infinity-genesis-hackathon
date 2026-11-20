"""Versioned constants shared by the generator, environment and grader."""

from decimal import Decimal

ENV_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
POLICY_VERSION = "1.0"

ISSUE_TYPES = (
    "duplicate_invoice",
    "payment_allocation",
    "fx_conversion",
    "period_cutoff",
)

SCENARIOS = (
    "partial_fx_payment_at_boundary",
    "duplicate_with_misapplied_payment",
)

# ISO-4217 minor-unit exponents of every currency the generator may use.
CURRENCY_EXPONENTS = {
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "CHF": 2,
    "CNY": 2,
    "KZT": 2,
    "JPY": 0,
    "KWD": 3,
}

# Rough USD value of one major unit; only used to make synthetic rates plausible.
USD_VALUE = {
    "USD": Decimal("1"),
    "EUR": Decimal("1.08"),
    "GBP": Decimal("1.27"),
    "CHF": Decimal("1.12"),
    "CNY": Decimal("0.138"),
    "KZT": Decimal("0.0021"),
    "JPY": Decimal("0.0066"),
    "KWD": Decimal("3.25"),
}

ACCOUNTS = {
    "1000": "Cash at bank",
    "1100": "Accounts receivable",
    "2300": "Unapplied customer receipts",
    "4000": "Revenue",
    "6100": "Rent expense",
    "6200": "Payroll expense",
    "6300": "Software expense",
    "6400": "Travel expense",
    "6500": "Utilities expense",
    "7900": "Realised FX gain/loss",
}
ACC_CASH = "1000"
ACC_AR = "1100"
ACC_UNAPPLIED = "2300"
ACC_REVENUE = "4000"
ACC_FX = "7900"
EXPENSE_ACCOUNTS = ("6100", "6200", "6300", "6400", "6500")

TABLES = ("ledger", "invoices", "payments", "fx")

MAX_STEPS = 100
PAGE_LIMIT = 50
MAX_FINDINGS = 30
MAX_EVIDENCE = 10
MAX_AFFECTED = 10
MAX_EXPLANATION = 1000
MAX_ID_LEN = 64
MAX_ERROR_MESSAGE = 200
MAX_EXPRESSION = 300
STEP_COST = Decimal("0.002")
TIME_LIMIT_S = 60.0

AMOUNT_TOL_ABS_MINOR = 2
AMOUNT_TOL_REL = Decimal("0.001")
