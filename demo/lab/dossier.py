"""The seven blocks, rendered as they complete.

Each step is written the moment it is declared, with the number it produced and
the method that produced it, rather than buffered until the run ends.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

STEPS = (
    ("assess", "ASSESS"),
    ("notify", "NOTIFY"),
    ("investigate", "INVESTIGATE"),
    ("decide", "DECIDE"),
    ("verify", "VERIFY"),
    ("close", "CLOSE"),
    ("remember", "REMEMBER"),
)

WIDTH = 74
DIM, BOLD, AMBER, GREEN, RESET = "\033[2m", "\033[1m", "\033[33m", "\033[32m", "\033[0m"


@dataclass
class Block:
    key: str
    label: str
    lines: list[str] = field(default_factory=list)
    method: str = ""
    seconds: float | None = None
    waiting: str = ""


@dataclass
class Dossier:
    table: str
    attempt: int
    stream = sys.stdout
    blocks: dict[str, Block] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    def open(self) -> None:
        title = f"ON-CALL AGENT · table {self.table} · run {self.attempt}"
        self._write(f"\n  {BOLD}{title}{RESET}{DIM}{'incident #4417'.rjust(WIDTH - len(title))}{RESET}\n")

    def step(self, key: str, *lines: str, method: str = "", seconds: float | None = None) -> None:
        """Land one step. Printed immediately - this is the whole point."""
        label = dict(STEPS).get(key, key.upper())
        block = Block(key=key, label=label, lines=[str(x) for x in lines], method=method, seconds=seconds)
        self.blocks[key] = block
        self._order.append(key)
        self._render(block)

    def waiting(self, key: str, prompt: str, command: str) -> None:
        """A gate. The run stops here until a person says a word.

        Keyed by whatever is pending, which may be several tools at once and may
        not correspond to a step that has been reported yet.
        """
        if key in self.blocks:
            self.blocks[key].waiting = prompt
        self._write(f"     {AMBER}||  {prompt}{RESET}\n")
        self._write(f"        {DIM}→ {command}{RESET}\n")

    def outcome(self, minutes_down: int, customers: int, unreconciled: int, woken: int,
                agreement: str) -> None:
        self._write(f"\n  {DIM}{'─' * WIDTH}{RESET}\n")
        cells = (
            f"{minutes_down} min down",
            f"{customers:,} customers turned away",
            f"{unreconciled:,} payments unreconciled" if unreconciled else "nothing unreconciled",
            f"{woken} woken" if woken else "nobody woken",
        )
        self._write("   " + f" {DIM}·{RESET} ".join(f"{BOLD}{c}{RESET}" for c in cells) + "\n")
        self._write(f"   {DIM}agreement with the room{RESET}  {GREEN}{agreement}{RESET}\n")

    def least_sure(self, sentence: str, path: str) -> None:
        """The line that sends a participant to the next file. Not decoration."""
        self._write(f"\n   {DIM}Least sure about{RESET}  {sentence}\n")
        self._write(f"   {DIM}{'':17}{RESET}  {BOLD}{path}{RESET}\n\n")

    def _render(self, block: Block) -> None:
        n = len(self._order)
        head = f"  {DIM}{n}{RESET}  {BOLD}{block.label:<12}{RESET}"
        first = block.lines[0] if block.lines else ""
        timing = f"{block.seconds:.1f}s" if block.seconds is not None else ""
        pad = max(1, WIDTH - len(block.label) - len(first) - len(timing) - 4)
        self._write(f"{head}{first}{' ' * pad}{DIM}{timing}{RESET}\n")
        for line in block.lines[1:]:
            self._write(f"     {' ' * 12}{line}\n")
        if block.method:
            self._write(f"     {' ' * 12}{DIM}method: {block.method}{RESET}\n")

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()
