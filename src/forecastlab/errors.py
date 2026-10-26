# src/forecastlab/errors.py
"""Tipe galat dan peringatan yang dipakai di seluruh paket."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Issue:
    """Satu galat (kode E-xx) atau peringatan (kode W-xx) untuk pengguna."""

    code: str
    message: str


class ForecastLabError(Exception):
    """Galat yang aman ditampilkan ke pengguna (tanpa stack trace)."""

    def __init__(self, issues: Issue | list[Issue]) -> None:
        self.issues: list[Issue] = [issues] if isinstance(issues, Issue) else list(issues)
        super().__init__("; ".join(f"[{i.code}] {i.message}" for i in self.issues))

    @property
    def code(self) -> str:
        return self.issues[0].code