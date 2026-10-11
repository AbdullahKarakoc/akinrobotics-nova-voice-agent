"""SQLite-backed tools: order status, product price/stock, service centers."""

from __future__ import annotations

import difflib
import re
import sqlite3
from pathlib import Path

from app.tools.registry import ToolRegistry
from app.tools.routing import extract_city, fold, mentions

STATUS_TEXT = {
    "preparing": ("hazırlanıyor", "being prepared"),
    "in_production": ("üretimde", "in production"),
    "shipped": ("kargoya verildi, yolda", "shipped, on the way"),
    "delivered": ("teslim edildi", "delivered"),
    "cancelled": ("iptal edildi", "cancelled"),
}

_ORDER_NO = re.compile(r"\b\d(?:[\d.\s]{1,6}\d)?\b")


def ensure_database(db_path: Path, seed_path: Path) -> None:
    if db_path.exists():
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(seed_path.read_text(encoding="utf-8"))


def register(registry: ToolRegistry, db_path: Path) -> None:
    def query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute(sql, params).fetchall()

    products = query("SELECT * FROM products")
    centers = query("SELECT city FROM service_centers")

    def match_product(text: str) -> sqlite3.Row | None:
        key = fold(text)
        words = set(re.findall(r"\w+", key))
        for r in products:
            if fold(r["code"]) in words or fold(r["name"]) in key:
                return r
        for r in products:
            if words & {fold(a) for a in r["aliases"].split(",")}:
                return r
        names = {fold(r["name"]): r for r in products}
        close = difflib.get_close_matches(key, names, n=1, cutoff=0.6)
        return names[close[0]] if close else None

    # ------------------------------------------------------------- orders
    def route_order(text: str) -> dict | None:
        if not mentions(text, r"sipari|order"):
            return None
        for m in _ORDER_NO.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 3 <= len(digits) <= 7:
                return {"order_number": digits}
        return None

    @registry.tool(
        "get_order_status",
        "Look up the status and estimated delivery date of a customer order by its order number.",
        {"order_number": {"type": "string", "description": "Order number, e.g. 1002"}},
        router=route_order,
    )
    def get_order_status(order_number: str) -> dict:
        digits = re.sub(r"\D", "", str(order_number))
        if not digits:
            return {"error": "No order number given. Ask the user for their order number."}
        rows = query(
            "SELECT o.*, p.name AS product FROM orders o JOIN products p ON p.code = o.product_code WHERE order_no = ?",
            (int(digits),),
        )
        if not rows:
            return {"found": False, "order_number": digits, "note": "No order with this number exists."}
        r = rows[0]
        tr, en = STATUS_TEXT.get(r["status"], (r["status"], r["status"]))
        return {
            "found": True,
            "order_number": r["order_no"],
            "product": r["product"],
            "quantity": r["quantity"],
            "status_tr": tr,
            "status_en": en,
            "order_date": r["order_date"],
            "estimated_delivery": r["eta"],
        }

    # ----------------------------------------------------------- products
    def route_product(text: str) -> dict | None:
        if not mentions(text, r"fiyat|ucret|kac para|kaca|kac lira|stok|satin al|teslim|price|cost|stock|buy|deliver"):
            return None
        product = match_product(text)
        return {"product": product["name"]} if product else None

    @registry.tool(
        "get_product_info",
        "Get the current list price (Turkish lira, VAT excluded), stock quantity and delivery time of a product.",
        {"product": {"type": "string", "description": "Product name or code, e.g. 'Rehber R2' or 'T5'"}},
        router=route_product,
    )
    def get_product_info(product: str) -> dict:
        match = match_product(product)
        if match is None:
            return {"found": False, "note": "Unknown product.", "available_products": [r["name"] for r in products]}
        in_stock = match["stock"] > 0
        return {
            "found": True,
            "product": match["name"],
            "price_try": match["price_try"],
            "price_note": "VAT excluded",
            "stock": match["stock"],
            "delivery": "5 business days" if in_stock else f"{match['lead_time_weeks']} weeks",
        }

    # ---------------------------------------------------- service centers
    def route_service(text: str) -> dict | None:
        if not mentions(text, r"servis|service cent|repair cent|tamir"):
            return None
        city = extract_city(text)
        return {"city": city} if city else None

    @registry.tool(
        "find_service_center",
        "Find the Nova Robotik technical service center address and phone number in a city.",
        {"city": {"type": "string", "description": "City name, e.g. İzmir"}},
        router=route_service,
    )
    def find_service_center(city: str) -> dict:
        rows = query("SELECT * FROM service_centers")
        key = fold(city)
        for r in rows:
            if fold(r["city"]) in key or key in fold(r["city"]):
                return {"found": True, "city": r["city"], "address": r["address"], "phone": r["phone"]}
        return {
            "found": False,
            "city": city,
            "note": "No service center in this city; on-site service is still available.",
            "cities_with_centers": [r["city"] for r in centers],
        }
