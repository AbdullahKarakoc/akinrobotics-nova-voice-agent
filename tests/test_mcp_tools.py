import json
from datetime import date

import pytest

from app.config import MCPServerConfig
from app.llm import ToolCall
from app.tools import mcp_tools, wikipedia
from app.tools.registry import ToolRegistry
from mcp_servers.nova_service import server

NOVA_SERVICE = "python -m mcp_servers.nova_service"


@pytest.fixture
def fixed_today(monkeypatch):
    monkeypatch.setattr(server, "today", lambda: date(2026, 10, 11))


def test_warranty_standard_extended_and_expired(fixed_today):
    new = server.check_warranty("R2-24011")
    new = server.check_warranty("R2-24011")
    assert new["warranty_en"] == "UNDER warranty: standard warranty valid until 2028-08-04"
    assert new["maintenance_en"] == "next maintenance on 2027-02-04 (last done 2026-08-04)"
    old = server.check_warranty("T5 21 507")  # spoken / split by STT
    assert old["warranty_tr"] == "garanti YOK: garanti süresi 2026-05-20 tarihinde doldu"
    assert old["battery_warranty_en"] == "no, it expired on 2025-05-20"
    assert old["maintenance_en"].startswith("maintenance overdue: it was due on 2026-05-18")
    assert server.check_warranty("22003")["warranty_en"] == "UNDER warranty: extended warranty valid until 2028-11-15"
    kit = server.check_warranty("E1-25140")  # 1-year warranty, no preventive maintenance
    assert kit["warranty_en"] == "NOT under warranty: it expired on 2026-09-01" and "maintenance_en" not in kit
    assert server.check_warranty("99999")["found"] is False
    assert server.check_warranty("12")["found"] is False


def test_add_months_clamps_to_month_end():
    assert server.add_months(date(2026, 8, 31), 6) == date(2027, 2, 28)
    assert server.add_months(date(2025, 11, 15), 2) == date(2026, 1, 15)


def test_diagnostics_and_tickets():
    d = server.get_robot_diagnostics("24012")
    assert d["connection_en"] == "offline (last seen 2026-10-10 22:15)" and d["battery_percent"] == 4
    assert d["active_error"]["code"] == "E-101"
    assert server.get_robot_diagnostics("C3-23088")["active_error"] is None
    t = server.get_service_ticket_status("5002")
    assert t["status_en"] == "a technician has been assigned" and t["technician_visit"] == "2026-10-12"
    assert "resolution_tr" not in t  # empty fields are dropped
    assert server.get_service_ticket_status("1234")["found"] is False
    assert server.get_service_ticket_status("")["found"] is False


def test_mcp_routers_fire_only_on_clear_intents():
    reg = ToolRegistry()
    for name, router in mcp_tools.ROUTERS.items():
        reg.tool(name, "", {}, router=router)(lambda: None)

    def routed(text):
        return [(c.name, c.arguments) for c in reg.route(text)]

    assert routed("24011 seri numaralı robotun garantisi devam ediyor mu?") == [("check_warranty", {"serial_number": "24011"})]
    assert routed("21507 seri numaralı robotumun garantisi devam ediyor mu?") == [
        ("check_warranty", {"serial_number": "21507"})
    ]
    assert routed("24012 numaralı robotum çalışmıyor.") == [("get_robot_diagnostics", {"serial_number": "24012"})]
    assert routed("5003 numaralı servis talebim ne durumda?") == [("get_service_ticket_status", {"ticket_number": "5003"})]
    assert routed("R2 24.011 garanti durumu nedir?") == [("check_warranty", {"serial_number": "24011"})]
    assert routed("When is the next maintenance of robot 21507?") == [("check_warranty", {"serial_number": "21507"})]
    assert routed("24012 numaralı robotum çalışmıyor") == [("get_robot_diagnostics", {"serial_number": "24012"})]
    assert routed("Robot 24012 ne durumda?") == [("get_robot_diagnostics", {"serial_number": "24012"})]
    assert routed("What is the status of robot 24012?") == [("get_robot_diagnostics", {"serial_number": "24012"})]
    assert routed("5002 numaralı servis talebim ne oldu?") == [("get_service_ticket_status", {"ticket_number": "5002"})]
    assert routed("Arıza kaydı 5003 ne durumda?") == [("get_service_ticket_status", {"ticket_number": "5003"})]
    # no number, or a knowledge question -> left to the LLM
    assert routed("Garanti süresi ne kadar?") == []
    assert routed("Robotumun bataryası kaç saat dayanır?") == []
    assert routed("Servis talebi nasıl açarım?") == []
    assert routed("1003 numaralı siparişim ne durumda?") == []  # 4 digits: an order, not a robot


async def test_agent_uses_nova_service_over_stdio():
    reg = ToolRegistry()
    mcp_tools.register(reg, [NOVA_SERVICE], timeout_s=60)
    try:
        assert {"check_warranty", "get_robot_diagnostics", "get_service_ticket_status"} <= set(reg.tools)
        schema = reg.tools["check_warranty"].schema()["function"]["parameters"]
        assert schema["required"] == ["serial_number"] and "title" not in schema["properties"]["serial_number"]

        (call,) = reg.route("21507 seri numaralı robotun garantisi bitti mi?")
        result = await reg.execute(call)
        data = json.loads(result.content)
        assert result.ok and data["product"] == "Taşıyıcı T5" and "2026-05-20" in data["warranty_en"]

        # arguments the schema does not declare are dropped; missing ones come back as a tool error
        ok = await reg.execute(ToolCall(id="a", name="get_service_ticket_status", arguments={"ticket_number": "5001", "x": 1}))
        assert json.loads(ok.content)["status_en"] == "resolved"
        bad = await reg.execute(ToolCall(id="b", name="get_robot_diagnostics", arguments={}))
        assert not bad.ok and "error" in json.loads(bad.content)
    finally:
        reg.close()


def test_unavailable_server_is_skipped():
    reg = ToolRegistry()
    mcp_tools.register(reg, ["python -m mcp_servers.does_not_exist"], timeout_s=30)
    assert reg.tools == {}


async def test_server_config_allowlist_prefix_and_router_only():
    reg = ToolRegistry()
    cfg = MCPServerConfig(target=NOVA_SERVICE, tools=["check_warranty"], prefix="x_", llm_visible=False, max_chars=30)
    mcp_tools.register(reg, [cfg], timeout_s=60)
    try:
        assert list(reg.tools) == ["x_check_warranty"]
        assert reg.schemas() == []  # router-only: never offered to the LLM
        result = await reg.execute(ToolCall(id="c", name="x_check_warranty", arguments={"serial_number": "24011"}))
        assert all(len(v) <= 34 for v in json.loads(result.content).values() if isinstance(v, str))
    finally:
        reg.close()


def test_trim_text_cuts_at_sentence_end():
    text = "LIDAR measures distance with laser pulses. It is used in robots. " * 5
    assert mcp_tools.trim_text({"summary": text, "n": 3}, 80) == {
        "summary": "LIDAR measures distance with laser pulses. It is used in robots.",
        "n": 3,
    }
    assert mcp_tools.trim_text("short", 80) == "short"


def test_wikipedia_routers_explain_glossary_terms_only():
    reg = ToolRegistry()
    for name, router in wikipedia.ROUTERS.items():
        reg.tool(name, "", {}, router=router)(lambda: None)

    def routed(text):
        return [(c.name, c.arguments) for c in reg.route(text)]

    assert routed("LiDAR nedir?") == [("wikipedia_tr_get_summary", {"title": "LIDAR"})]
    assert routed("Yapay zeka ne demek?") == [("wikipedia_tr_get_summary", {"title": "Yapay zekâ"})]
    assert routed("Ultrasonik sensör nasıl çalışır?") == [("wikipedia_en_get_summary", {"title": "Ultrasonic sensor"})]
    assert routed("SLAM nedir?") == [("wikipedia_en_get_summary", {"title": "Simultaneous localization and mapping"})]
    assert routed("What does lidar mean?") == [("wikipedia_en_get_summary", {"title": "Lidar"})]
    assert routed("What is a humanoid robot?") == [("wikipedia_en_get_summary", {"title": "Humanoid robot"})]
    # our products are answered from the documents; off-topic and non-questions never reach the web
    assert routed("Rehber R2'nin sensörleri nedir?") == []
    assert routed("Robotunuzda LiDAR var mı?") == []
    assert routed("Türkiye'nin başkenti nedir?") == []
    assert routed("Who won the last football World Cup?") == []
    assert routed("Garanti süresi nedir?") == []
    assert routed("Servis ücreti nedir?") == []
