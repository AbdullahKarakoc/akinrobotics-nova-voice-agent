"""Nova Servis: an MCP server for Nova Robotik's after-sales system.

It stands for a system owned by another team (warranty records, remote robot telemetry,
service tickets) that the voice agent reaches over the Model Context Protocol instead of
importing its code. Any MCP client can use it, e.g. MCP Inspector or Claude Desktop.

All tools are read-only. Answers are compact JSON with Turkish and English texts, like the
agent's local tools, and an unknown number returns `found: false` instead of an error so the
model tells the user rather than guessing.
"""

from __future__ import annotations

import calendar
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

SEED = Path(__file__).with_name("seed.sql")
BATTERY_WARRANTY_MONTHS = 12
MAINTENANCE_INTERVAL_MONTHS = 6

TICKET_STATUS = {
    "open": ("açık, bir teknisyen atanması bekleniyor", "open, waiting for a technician to be assigned"),
    "technician_assigned": ("teknisyen atandı", "a technician has been assigned"),
    "waiting_parts": ("yedek parça bekleniyor", "waiting for spare parts"),
    "resolved": ("çözüldü", "resolved"),
}

mcp = MCPServer(
    "nova-service",
    instructions="Nova Robotik after-sales data: warranty, robot diagnostics and service tickets.",
    log_level="WARNING",
)

_db = sqlite3.connect(":memory:", check_same_thread=False)
_db.row_factory = sqlite3.Row
_db.executescript(SEED.read_text(encoding="utf-8"))


def today() -> date:
    return date.today()


def add_months(d: date, months: int) -> date:
    year, month = divmod(d.month - 1 + months, 12)
    year, month = d.year + year, month + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return _db.execute(sql, params).fetchall()


def _digits(text: str) -> str:
    return re.sub(r"\D", "", str(text))


def _robot(serial_number: str) -> sqlite3.Row | None:
    """Matches 'R2-24011', '24011', 'R2 24 011' ...: the five digits identify a robot."""
    digits = _digits(serial_number)
    if len(digits) < 5:
        return None
    rows = _query(
        "SELECT r.*, m.name AS product, m.warranty_months FROM robots r JOIN models m ON m.code = r.model_code "
        "WHERE substr(r.serial, -5) = ?",
        (digits[-5:],),
    )
    return rows[0] if rows else None


def _not_found(serial_number: str) -> dict[str, Any]:
    return {"found": False, "serial_number": serial_number, "note": "No robot with this serial number exists."}


SerialNumber = Annotated[str, Field(description="Robot serial number, e.g. R2-24011 (the 5 digits are enough)")]


@mcp.tool()
def check_warranty(serial_number: SerialNumber) -> dict[str, Any]:
    """Warranty and preventive-maintenance status of a delivered robot, by its serial number."""
    robot = _robot(serial_number)
    if robot is None:
        return _not_found(serial_number)
    # Verdicts are spelled out, verdict first: a 4B model read `"in_warranty": false`, and even
    # "expired on <date>", as "still covered" (it does not know today's date).
    now = today()
    delivered = date.fromisoformat(robot["delivered"])
    extended = robot["extended_until"]
    until = date.fromisoformat(extended) if extended else add_months(delivered, robot["warranty_months"])
    battery_until = add_months(delivered, BATTERY_WARRANTY_MONTHS)
    kind_tr, kind_en = ("uzatılmış garanti", "extended warranty") if extended else ("standart garanti", "standard warranty")
    result: dict[str, Any] = {
        "found": True,
        "serial_number": robot["serial"],
        "product": robot["product"],
        "delivered": robot["delivered"],
        "warranty_tr": f"garanti VAR: {kind_tr} {until} tarihine kadar geçerli"
        if now <= until else f"garanti YOK: garanti süresi {until} tarihinde doldu",
        "warranty_en": f"UNDER warranty: {kind_en} valid until {until}"
        if now <= until else f"NOT under warranty: it expired on {until}",
        "battery_warranty_tr": f"VAR, {battery_until} tarihine kadar" if now <= battery_until else f"YOK, {battery_until} tarihinde doldu",
        "battery_warranty_en": f"yes, until {battery_until}" if now <= battery_until else f"no, it expired on {battery_until}",
    }
    if robot["last_maintenance"]:
        last = robot["last_maintenance"]
        due = add_months(date.fromisoformat(last), MAINTENANCE_INTERVAL_MONTHS)
        if due < now:
            result["maintenance_tr"] = f"bakım gecikmiş: {due} tarihinde yapılmalıydı (son bakım {last})"
            result["maintenance_en"] = f"maintenance overdue: it was due on {due} (last done {last})"
        else:
            result["maintenance_tr"] = f"sıradaki bakım {due} (son bakım {last})"
            result["maintenance_en"] = f"next maintenance on {due} (last done {last})"
    return result


@mcp.tool()
def get_robot_diagnostics(serial_number: SerialNumber) -> dict[str, Any]:
    """Live remote diagnostics of a robot by its serial number: online or not, battery level,
    software version and any active error with what the customer can do about it."""
    robot = _robot(serial_number)
    if robot is None:
        return _not_found(serial_number)
    result: dict[str, Any] = {
        "found": True,
        "serial_number": robot["serial"],
        "product": robot["product"],
        "connection_tr": "çevrimiçi" if robot["online"] else f"çevrimdışı (son bağlantı {robot['last_seen']})",
        "connection_en": "online" if robot["online"] else f"offline (last seen {robot['last_seen']})",
        "battery_percent": robot["battery_pct"],
        "software_version": robot["firmware"],
        "active_error": None,
    }
    if robot["error_code"]:
        err = _query("SELECT * FROM error_codes WHERE code = ?", (robot["error_code"],))[0]
        result["active_error"] = {k: err[k] for k in ("code", "issue_tr", "issue_en", "advice_tr", "advice_en")}
    return result


@mcp.tool()
def get_service_ticket_status(
    ticket_number: Annotated[str, Field(description="Service request (ticket) number, e.g. 5002")],
) -> dict[str, Any]:
    """Status of a technical service request (repair ticket) by its ticket number."""
    digits = _digits(ticket_number)
    rows = _query(
        "SELECT t.*, m.name AS product FROM tickets t JOIN robots r ON r.serial = t.serial "
        "JOIN models m ON m.code = r.model_code WHERE ticket_no = ?",
        (int(digits),),
    ) if digits else []
    if not rows:
        return {"found": False, "ticket_number": ticket_number, "note": "No service request with this number exists."}
    t = rows[0]
    tr, en = TICKET_STATUS.get(t["status"], (t["status"], t["status"]))
    result = {
        "found": True,
        "ticket_number": t["ticket_no"],
        "serial_number": t["serial"],
        "product": t["product"],
        "opened": t["opened"],
        "issue_tr": t["issue_tr"],
        "issue_en": t["issue_en"],
        "status_tr": tr,
        "status_en": en,
        "technician_visit": t["technician_visit"],
        "expected_resolution": t["expected_resolution"],
        "resolution_tr": t["resolution_tr"],
        "resolution_en": t["resolution_en"],
    }
    return {k: v for k, v in result.items() if v is not None}  # every token is prompt prefill time
