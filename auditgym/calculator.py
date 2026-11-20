"""Arithmetic-only calculator: decimal numbers, + - * /, unary signs and parentheses.

Hand-written tokenizer + recursive-descent parser (no eval, no ast, no names,
no calls, no attribute access, no strings). Exact Decimal arithmetic, 34 digits.
"""

from __future__ import annotations

import re
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext

from .constants import MAX_EXPRESSION

TOKEN_RE = re.compile(r"\s*(?:(\d+(?:\.\d*)?|\.\d+)|(\S))")
MAX_DEPTH = 40


class CalcError(ValueError):
    pass


def _tokens(expr: str):
    out = []
    pos = 0
    while pos < len(expr):
        m = TOKEN_RE.match(expr, pos)
        if not m or m.end() == pos:
            break
        num, op = m.group(1), m.group(2)
        if num is not None:
            out.append(("num", num))
        elif op is not None:
            if op not in "+-*/()":
                raise CalcError("only numbers, + - * / and parentheses are allowed")
            out.append(("op", op))
        pos = m.end()
    out.append(("end", None))
    return out


class _Parser:
    def __init__(self, toks):
        self.t = toks
        self.i = 0
        self.depth = 0

    def peek(self):
        return self.t[self.i]

    def take(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expr(self):
        v = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            r = self.term()
            v = v + r if op == "+" else v - r
        return v

    def term(self):
        v = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            r = self.unary()
            if op == "*":
                v = v * r
            else:
                if r == 0:
                    raise CalcError("division by zero")
                v = v / r
        return v

    def unary(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise CalcError("expression too deeply nested")
        try:
            tok = self.peek()
            if tok in (("op", "+"), ("op", "-")):
                self.take()
                v = self.unary()
                return v if tok[1] == "+" else -v
            return self.atom()
        finally:
            self.depth -= 1

    def atom(self):
        tok = self.take()
        if tok[0] == "num":
            return Decimal(tok[1])
        if tok == ("op", "("):
            self.depth += 1
            if self.depth > MAX_DEPTH:
                raise CalcError("expression too deeply nested")
            v = self.expr()
            self.depth -= 1
            if self.take() != ("op", ")"):
                raise CalcError("missing closing parenthesis")
            return v
        raise CalcError("syntax error")


def evaluate(expression) -> str:
    if not isinstance(expression, str):
        raise CalcError("expression must be a string")
    if not expression.strip():
        raise CalcError("empty expression")
    if len(expression) > MAX_EXPRESSION:
        raise CalcError("expression longer than %d characters" % MAX_EXPRESSION)
    with localcontext() as ctx:
        ctx.prec = 34
        ctx.Emax = 999
        ctx.Emin = -999
        ctx.traps[InvalidOperation] = True
        ctx.traps[DivisionByZero] = True
        try:
            p = _Parser(_tokens(expression.strip()))
            v = p.expr()
            if p.peek()[0] != "end":
                raise CalcError("syntax error")
            v = +v
        except CalcError:
            raise
        except (InvalidOperation, DivisionByZero, OverflowError, ArithmeticError):
            raise CalcError("arithmetic error")
    s = format(v.normalize(), "f") if v == v.to_integral_value() else format(v, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    if len(s) > 120:
        raise CalcError("result too large")
    return s
