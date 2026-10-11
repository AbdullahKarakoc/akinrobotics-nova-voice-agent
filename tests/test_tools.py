import json
from pathlib import Path

import pytest

from app.llm import ToolCall
from app.tools import database, web_apis
from app.tools.registry import ToolRegistry
from app.tools.routing import extract_city

SEED = Path(__file__).resolve().parent.parent / "data" / "seed.sql"


@pytest.fixture
def registry(tmp_path):
    db = tmp_path / "test.db"
    database.ensure_database(db, SEED)
    reg = ToolRegistry()
    database.register(reg, db)
    web_apis.register(reg)
    return reg


async def call(reg: ToolRegistry, name: str, **args) -> dict:
    result = await reg.execute(ToolCall(id="t", name=name, arguments=args))
    return json.loads(result.content)


async def test_order_status_accepts_spoken_number_formats(registry):
    for spoken in ("340", "NR-340", "3 40"):
        r = await call(registry, "get_order_status", order_number=spoken)
        assert r["found"] and r["product"] == "Temizlik C3" and r["status_en"].startswith("shipped")
    assert (await call(registry, "get_order_status", order_number="999"))["found"] is False


async def test_product_lookup_by_name_code_and_alias(registry):
    assert (await call(registry, "get_product_info", product="T5"))["price_try"] == 890000
    assert (await call(registry, "get_product_info", product="taşıyıcı t5"))["delivery"] == "5 weeks"
    assert (await call(registry, "get_product_info", product="the cleaning robot"))["product"] == "Temizlik C3"
    assert (await call(registry, "get_product_info", product="drone"))["found"] is False


async def test_service_center_is_accent_insensitive(registry):
    assert (await call(registry, "find_service_center", city="izmir"))["phone"].startswith("0232")
    assert (await call(registry, "find_service_center", city="Konya"))["found"] is False


async def test_tool_errors_are_reported_not_raised(registry):
    result = await registry.execute(ToolCall(id="x", name="does_not_exist", arguments={}))
    assert not result.ok and "error" in json.loads(result.content)


def test_routers_fire_only_on_clear_intents(registry):
    def routed(text):
        return [(c.name, c.arguments) for c in registry.route(text)]

    assert routed("340 numaralı siparişim ne durumda?") == [("get_order_status", {"order_number": "340"})]
    assert routed("Taşıyıcı T5'in fiyatı ne kadar?") == [("get_product_info", {"product": "Taşıyıcı T5"})]
    assert routed("How much does the education kit cost?") == [("get_product_info", {"product": "Eğitim E1"})]
    assert routed("İzmir'de servis merkeziniz var mı?") == [("find_service_center", {"city": "İzmir"})]
    assert routed("Ankara'da hava nasıl?") == [("get_weather", {"city": "Ankara"})]
    assert routed("Ankara'da havanası") == [("get_weather", {"city": "Ankara"})]  # STT glued the words
    assert routed("Ankara'da havan nasıl?") == [("get_weather", {"city": "Ankara"})]
    assert routed("What time is it?") == [("get_current_datetime", {})]
    assert routed("Saat kaç?") == [("get_current_datetime", {})]
    # knowledge questions and ambiguous ones go to the LLM
    assert routed("Rehber R2'nin pil ömrü ne kadar?") == []
    assert routed("How much does the Rehber R2 weigh?") == []
    assert routed("Teslimat ne kadar sürer?") == []
    assert routed("Hava nasıl?") == []  # no city -> let the model ask
    assert routed("Siparişim ne durumda?") == []  # no number -> let the model ask


def test_extract_city():
    assert extract_city("Eskişehir'de servis var mı") == "Eskişehir"
    assert extract_city("Londra'da hava nasıl") == "Londra"
    assert extract_city("weather in Berlin today") == "Berlin"
    assert extract_city("hava nasıl") is None
