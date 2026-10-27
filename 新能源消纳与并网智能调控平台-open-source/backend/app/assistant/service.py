from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any
import html
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from app.adapters.stations import StationRepository
from app.config import get_settings
from app.services.metrics import calculate_consumption
from app.services.time_alignment import CN_TZ, classify_freshness
from app.rag_config import load_backend_env

# Tests and packaged launches can resolve ``app`` from an installed egg-info
# path instead of the backend working directory. Make the bundled RAG package
# importable without requiring a separate pip installation.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = _BACKEND_ROOT.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from rag_core.power_eval.cloud_llm import ChatCloudLLM
from rag_core.power_eval.config import PowerEvalConfig
from rag_core.power_eval.corpus import PowerDocument, load_knowledge_corpus
from rag_core.power_eval.disco_pipeline import CloudDiscoRAGPipeline
from rag_core.power_eval.index import SearchHit, VectorIndex
from rag_core.power_eval.llm_client import DeepSeekClient
from rag_core.src.rst_tree.cache import RstCache


@dataclass(frozen=True)
class AssistantResult:
    answer: str
    sources: list[dict[str, Any]]
    steps: list[dict[str, str]]
    live_context: dict[str, Any]
    mode: str
    confidence: str


_index: VectorIndex | None = None
_documents: list[PowerDocument] = []
_load_error = ""
_index_lock = Lock()


class _EmptyIndex:
    """Placeholder when the private knowledge corpus has not been installed."""

    documents: list[PowerDocument] = []

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        return []


def _get_index() -> tuple[VectorIndex | _EmptyIndex, list[PowerDocument]]:
    global _index, _documents, _load_error
    if _index is None:
        with _index_lock:
            if _index is None:
                root = Path(__file__).resolve().parents[2] / "data" / "knowledge"
                try:
                    _documents = load_knowledge_corpus(root)
                    if not _documents:
                        _load_error = f"Knowledge corpus is empty: {root}"
                        _index = _EmptyIndex()  # type: ignore[assignment]
                    else:
                        _index = VectorIndex(_documents)
                except Exception as exc:
                    _load_error = str(exc)
                    _documents = []
                    _index = _EmptyIndex()  # type: ignore[assignment]
    return _index, _documents


def _disco_knowledge(question: str, index: VectorIndex, settings: PowerEvalConfig):
    """Run the documented Disco-RAG structure path for the knowledge agent."""
    if settings.rag_mode != "disco" or not settings.api_key:
        return None
    cache_path = Path(settings.rst_cache) if settings.rst_cache else None
    if cache_path is not None and not cache_path.is_absolute():
        cache_path = _BACKEND_ROOT / cache_path
    cache = RstCache.load(cache_path) if cache_path is not None else RstCache()
    llm = ChatCloudLLM(DeepSeekClient(settings), settings.model, output_language="Chinese")
    pipeline = CloudDiscoRAGPipeline(
        index, llm, top_k=settings.top_k, rst_cache=cache, require_rst_cache=False,
    )
    result = pipeline.run(question)
    if cache_path is not None:
        try:
            cache.save(cache_path)
        except OSError:
            pass
    return result


def _live_snapshot(repo: StationRepository, station_id: str) -> dict[str, Any]:
    station = next((item for item in repo.list_stations() if item.station_id == station_id), None)
    if station is None:
        return {"station_id": station_id, "available": False, "reason": "场站不存在"}
    snapshot = repo.latest_snapshot(station_id)
    weather = repo.latest_weather(station_id)
    if snapshot is None:
        return {"station_id": station_id, "available": False, "station_type": station.station_type, "weather": weather}
    now = datetime.now(timezone.utc)
    freshness = classify_freshness(snapshot.observed_at, now)
    consumption = calculate_consumption(
        snapshot.active_power_mw, snapshot.theory_power_mw, snapshot.available_power_mw, snapshot.limit_mark,
    )
    return {
        "station_id": station_id,
        "station_type": station.station_type,
        "available": True,
        "observed_at": snapshot.observed_at.isoformat(),
        "freshness": freshness,
        "active_power_mw": snapshot.active_power_mw,
        "theory_power_mw": snapshot.theory_power_mw,
        "available_power_mw": snapshot.available_power_mw,
        "limit_mark": snapshot.limit_mark,
        "weather": weather,
        "consumption": {
            "utilization_rate": consumption.utilization_rate,
            "curtailed_mw": consumption.curtailed_mw,
            "curtailment_rate": consumption.curtailment_rate,
        },
    }


def _live_context(repo: StationRepository, station_id: str | None) -> dict[str, Any]:
    if station_id:
        end = datetime.now(timezone.utc)
        power_rows = repo.power_series(station_id, end - timedelta(hours=4), end)[-16:]
        weather_rows = repo.weather_series(station_id, end - timedelta(hours=4), end)[-16:]
        forecast = repo.latest_forecast_window(station_id, 16)
        return {
            "scope": station_id,
            "stations": [_live_snapshot(repo, station_id)],
            "recent_power_4h": [
                {"time": row.observed_at.isoformat(), "actual_mw": row.active_power_mw, "theory_mw": row.theory_power_mw, "available_mw": row.available_power_mw}
                for row in power_rows
            ],
            "recent_weather_4h": [
                {key: value.isoformat() if isinstance(value, datetime) else value for key, value in row.items()}
                for row in weather_rows
            ],
            "ultra_short_power_forecast": {
                "issue_at": forecast.get("issue_at").isoformat() if isinstance(forecast.get("issue_at"), datetime) else forecast.get("issue_at"),
                "horizon_hours": forecast.get("horizon_hours"),
                "points": [
                    {key: value.isoformat() if isinstance(value, datetime) else value for key, value in row.items()}
                    for row in forecast.get("points", [])[-32:]
                ],
            },
            "nwp_forecast_4h": _station_nwp_context(station_id),
        }
    stations = repo.list_stations()
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(lambda item: _live_snapshot(repo, item.station_id), stations))
    online = sum(1 for row in rows if row.get("available"))
    total_active = sum((row.get("active_power_mw") or 0) for row in rows)
    total_theory = sum((row.get("theory_power_mw") or 0) for row in rows)
    total_available = sum((row.get("available_power_mw") or 0) for row in rows)
    aggregate = calculate_consumption(total_active, total_theory, total_available, None)
    def trend(item):
        end = datetime.now(timezone.utc)
        values = repo.power_series(item.station_id, end - timedelta(hours=1), end)[-5:]
        numbers = [float(row.active_power_mw) for row in values if row.active_power_mw is not None]
        return {"station_id": item.station_id, "points": len(numbers), "start_mw": round(numbers[0], 3) if numbers else None, "end_mw": round(numbers[-1], 3) if numbers else None, "min_mw": round(min(numbers), 3) if numbers else None, "max_mw": round(max(numbers), 3) if numbers else None}
    with ThreadPoolExecutor(max_workers=6) as pool:
        trends = list(pool.map(trend, stations))
    return {"scope": "ALL", "online": online, "total": len(rows), "total_active_power_mw": round(total_active, 3), "total_available_power_mw": round(total_available, 3),
            "consumption": {"utilization_rate": aggregate.utilization_rate, "curtailed_mw": aggregate.curtailed_mw, "curtailment_rate": aggregate.curtailment_rate},
            "stations": rows, "recent_power_1h_summary": trends}


def _station_nwp_context(station_id: str) -> dict[str, Any]:
    path = _PROJECT_ROOT / "data" / "weather" / "latest_station_nwp.json"
    if not path.exists():
        return {"available": False, "points": []}
    try:
        import json
        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped: dict[str, dict[str, Any]] = {}
        for row in payload.get("stations", []):
            if row.get("station_id") != station_id or not row.get("valid_time_utc"):
                continue
            valid_utc = str(row["valid_time_utc"])
            valid_cn = row.get("valid_time_cn")
            if not valid_cn:
                valid_cn = datetime.fromisoformat(valid_utc.replace("Z", "+00:00")).astimezone(CN_TZ).isoformat()
            point = grouped.setdefault(valid_cn, {"valid_time_utc": valid_utc, "valid_time_cn": valid_cn, "variables": {}})
            point["variables"].update(row.get("variables") or {})
        all_points = [grouped[key] for key in sorted(grouped)]
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=15)
        future_points = []
        for point in all_points:
            try:
                valid_time = datetime.fromisoformat(point["valid_time_utc"].replace("Z", "+00:00"))
                if valid_time >= cutoff:
                    future_points.append(point)
            except ValueError:
                continue
        points = future_points[:16] if future_points else all_points[-16:]
        return {"available": bool(points), "issued_at_utc": payload.get("issued_at_utc"), "issued_at_cn": payload.get("issued_at_cn"), "source_file": payload.get("source_file"), "points": points}
    except Exception:
        return {"available": False, "points": []}


def _is_live_question(question: str) -> bool:
    # The control-room assistant is a live operational agent.  A user asking
    # for an “运行态势分析报告” has implicitly requested the current
    # operating context even when they did not use the literal word “实时”.
    # Always load the latest power/weather/limit snapshot; this prevents the
    # model from producing a misleading “未请求实时数据” report.
    return True


def _control_objective(question: str) -> str:
    if any(word in question for word in ("限电", "弃电", "消纳")):
        return "消纳提升与限电处置"
    if any(word in question for word in ("并网", "电压", "无功", "频率")):
        return "并网安全与辅助服务"
    if any(word in question for word in ("预测", "误差", "准确")):
        return "预测偏差诊断"
    if any(word in question for word in ("天气", "风速", "辐照", "气象", "告警")):
        return "气象风险与运行预警"
    return "运行态势分析"


def _detect_intent(question: str) -> str:
    """Classify response intent so ordinary chat is not forced into a report."""
    if any(word in question for word in ("报告", "报表", "运行态势分析", "正式分析", "总结报告")):
        return "report"
    if any(word in question for word in ("为什么", "原因", "解释", "怎么回事", "差异")):
        return "explanation"
    if any(word in question for word in ("建议", "怎么做", "下一步", "如何处置", "调控动作", "措施")):
        return "recommendation"
    if any(word in question for word in ("多少", "多少兆瓦", "当前", "现在", "有没有", "是否", "状态", "查询", "看一下")):
        return "query"
    return "general"


def _control_plan(question: str, live: dict[str, Any]) -> dict[str, Any]:
    """Produce auditable, conservative actions when a cloud model is absent."""
    objective = _control_objective(question)
    rows = live.get("stations", [])
    actions: list[dict[str, str]] = []
    evidence: list[str] = []
    for row in rows:
        if not row.get("available"):
            continue
        station = row.get("station_id", "场站")
        actual = row.get("active_power_mw")
        available = row.get("available_power_mw")
        if available is not None and actual is not None:
            curtailed = max(0.0, float(available) - float(actual))
            if curtailed > 0.05:
                evidence.append(f"{station} 可利用功率与实际功率差 {curtailed:.2f} MW")
                actions.append({"station_id": station, "action": "核查调度限额、送出断面和AGC指令，确认可利用功率是否可释放", "priority": "high"})
        if row.get("limit_mark"):
            evidence.append(f"{station} 存在限电标记")
            actions.append({"station_id": station, "action": "保留限电标记证据，联动检查断面/电压/调度指令后再调整出力", "priority": "high"})
        weather = row.get("weather") or {}
        wind = weather.get("wind_speed_ms")
        if wind is not None and float(wind) >= 13.9:
            evidence.append(f"{station} 测风塔风速 {float(wind):.1f} m/s")
            actions.append({"station_id": station, "action": "按大风运行规程限制爬坡速率并复核机组安全边界，不建议直接追求出力最大化", "priority": "high"})
        ghi = weather.get("ghi_wm2")
        if ghi is not None and float(ghi) < 20 and row.get("station_type") == "pv":
            actions.append({"station_id": station, "action": "辐照度较低，保持并网状态监视，避免把低辐照误判为设备故障", "priority": "normal"})
    if not actions:
        actions.append({"station_id": live.get("scope", "全场"), "action": "维持当前调度策略，持续监视15分钟功率、限电标记和气象变化；出现连续偏差后再触发策略切换", "priority": "normal"})
    return {"objective": objective, "actions": actions[:8], "evidence": evidence[:12], "constraints": ["不越过调度下发限额", "不替代现场保护与调度指令", "策略执行前需核对最新功率和气象时间戳"]}


def _live_fallback(question: str, live: dict[str, Any]) -> str:
    rows = live.get("stations", [])
    if not rows:
        return "当前没有读取到实时场站数据，暂时无法形成调控建议。"
    if live.get("scope") == "ALL":
        return f"已读取 {live.get('online', 0)}/{live.get('total', 0)} 个场站，当前新能源实时功率合计 {live.get('total_active_power_mw', 0):.2f} MW。需要具体建议时，请指定场站或说明调控目标。"
    row = rows[0]
    if not row.get("available"):
        return f"{row.get('station_id')} 当前没有可用的实时快照，无法给出可靠的调控建议。"
    weather = row.get("weather") or {}
    parts = [f"{row['station_id']} 最新观测时间 {row.get('observed_at', '—')}，实时功率 {row.get('active_power_mw', '—')} MW"]
    if row.get("theory_power_mw") is not None:
        parts.append(f"理论功率 {row['theory_power_mw']} MW")
    if weather.get("wind_speed_ms") is not None:
        parts.append(f"测风塔风速 {weather['wind_speed_ms']} m/s")
    if weather.get("ghi_wm2") is not None:
        parts.append(f"辐照度 {weather['ghi_wm2']} W/m²")
    parts.append("以上为实时数据读取结果；如需策略建议，请补充调控目标、时间范围和约束条件。")
    return "，".join(str(item) for item in parts) + "。"


def _humanize_answer(text: str) -> str:
    """Remove raw Python/JSON field syntax before showing an answer to operators."""
    labels = {
        "observed_at": "观测时间", "active_power_mw": "实时功率", "theory_power_mw": "理论功率",
        "available_power_mw": "可利用功率", "limit_mark": "限电标记", "wind_speed_ms": "风速",
        "wind_direction_deg": "风向", "temperature_c": "温度", "humidity_pct": "湿度",
        "pressure_hpa": "气压", "ghi_wm2": "辐照度", "available": "数据可用", "freshness": "新鲜度",
    }
    cleaned = text
    for key, label in labels.items():
        # Handles `key=value`, key=value and key: value forms emitted by models.
        cleaned = re.sub(rf"`?{re.escape(key)}`?\s*[=:]\s*`?([^`;,\s]+)`?", rf"{label}：\1", cleaned)
        cleaned = re.sub(rf"`?{re.escape(key)}`?", label, cleaned)
    cleaned = re.sub(r"限电标记：\s*0\b", "限电标记：否", cleaned)
    cleaned = re.sub(r"限电标记：\s*1\b", "限电标记：是", cleaned)
    cleaned = re.sub(r"\bTrue\b", "是", cleaned)
    cleaned = re.sub(r"\bFalse\b", "否", cleaned)
    cleaned = re.sub(r"\bNone\b", "暂无", cleaned)
    cleaned = cleaned.replace("`", "")
    return cleaned


def _general_fallback(question: str, matched: bool = False) -> str:
    objective = _control_objective(question)
    prefix = "已检索到相关新能源调控知识，但当前未配置模型 API Key，以下为基于检索依据和通用调度原则的初步方案" if matched else "当前未命中专用知识库，以下为基于通用新能源调度原则的初步方案"
    return (f"{prefix}（目标：{objective}）：\n"
            "1. 先核对场站最新有功、理论功率、可利用功率、限电标记及数据时间戳，确认是否为真实受限而非数据异常。\n"
            "2. 若可利用功率高于实际功率，依次检查调度限额、送出断面、AGC/AVC指令、设备可用状态和通信链路。\n"
            "3. 若涉及并网安全，优先满足频率、电压、爬坡率和保护定值约束，不得越过调度下发限额。\n"
            "4. 调整后用连续 15 分钟功率、限电标记和气象观测复核效果，并记录动作前后指标。\n"
            "这是通用建议，不替代调度指令、保护逻辑和现场规程；接入具体场站实时数据后可进一步细化。")


def _web_search(question: str, limit: int = 5) -> list[dict[str, str]]:
    """Fetch public web result summaries without exposing credentials."""
    try:
        url = "https://www.bing.com/search?format=rss&setlang=zh-CN&cc=cn&q=" + urllib.parse.quote(question)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 NewEnergyControl/1.0"})
        with urllib.request.urlopen(request, timeout=8) as response:
            root = ET.fromstring(response.read())
        results: list[dict[str, str]] = []
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            raw_description = html.unescape(item.findtext("description") or "")
            description = re.sub(r"<[^>]+>", " ", raw_description)
            description = re.sub(r"\s+", " ", description).strip()
            if title and link:
                results.append({"title": title, "url": link, "snippet": description[:500]})
        return results
    except Exception:
        return []


def answer_question(question: str, repo: StationRepository, station_id: str | None = None, use_web_search: bool = False, history: list[dict[str, str]] | None = None) -> AssistantResult:
    load_backend_env()
    question = question.strip()
    if not question:
        return AssistantResult("请输入需要分析的问题。", [], [{"agent": "问题理解", "status": "waiting", "detail": "未收到问题"}], {}, "guardrail", "low")
    intent = _detect_intent(question)
    live_needed = _is_live_question(question)
    resolved_station_id = station_id
    if not resolved_station_id:
        resolved_station_id = next((item.station_id for item in repo.list_stations() if item.station_id.upper() in question.upper()), None)
    live = _live_context(repo, resolved_station_id) if live_needed else {}
    control = _control_plan(question, live)
    steps = [{"agent": "问题理解智能体", "status": "done", "detail": f"已形成初步意图：{intent}；最终回答以当前问题和对话语义为准"}]
    steps.append({"agent": "实时数据智能体", "status": "done", "detail": "已读取实时快照、历史功率/气象、超短期功率预测与未来 NWP"})
    forecast = live.get("ultra_short_power_forecast") or {}
    nwp = live.get("nwp_forecast_4h") or {}
    forecast_points = len(forecast.get("points") or [])
    steps.append({
        "agent": "预测分析智能体",
        "status": "done" if forecast_points or nwp.get("available") else "empty",
        "detail": (
            f"预测批次 {forecast.get('issue_at') or '—'}，{forecast_points} 个未来点；"
            f"NWP {'已接入' if nwp.get('available') else '未接入'}"
        ),
    })
    settings = PowerEvalConfig.from_env(require_api_key=False)
    index, _ = _get_index()
    hits = index.search(question, settings.top_k)
    useful_hits = [hit for hit in hits if hit.score >= 0.08]
    disco_result = None
    if useful_hits and settings.api_key and settings.rag_mode == "disco":
        try:
            disco_result = _disco_knowledge(question, index, settings)
            steps.append({
                "agent": "知识检索智能体",
                "status": "done",
                "detail": f"Disco-RAG 结构化证据：{len(disco_result.trees)} 个 RST 树、修辞图、话语规划",
            })
        except Exception as exc:
            steps.append({
                "agent": "知识检索智能体",
                "status": "fallback",
                "detail": f"Disco-RAG 结构阶段不可用，退回片段检索：{type(exc).__name__}",
            })
    if disco_result is None:
        steps.append({"agent": "知识检索智能体", "status": "done" if useful_hits else "empty", "detail": f"命中 {len(useful_hits)} 条相关知识" if useful_hits else "知识库没有检索到足够相关内容"})
    web_results = _web_search(question) if use_web_search else []
    if use_web_search:
        steps.append({"agent": "联网搜索智能体", "status": "done" if web_results else "empty", "detail": f"检索到 {len(web_results)} 条公开网页资料" if web_results else "本次联网搜索未返回可用结果"})
    source_rows = [{"source_id": hit.document.source_id, "score": round(hit.score, 4), "title": "知识库依据", "excerpt": hit.document.text[:360]} for hit in useful_hits]
    source_rows.extend({"source_id": item["url"], "score": 1.0, "title": item["title"], "excerpt": item["snippet"]} for item in web_results)
    if not settings.api_key:
        steps.append({"agent": "报告生成智能体", "status": "fallback", "detail": "未配置模型 API Key，使用实时数据与通用调控规则生成建议"})
        fallback = _live_fallback(question, live) if live_needed else _general_fallback(question, bool(useful_hits))
        if live_needed and control and intent in {"report", "recommendation", "explanation"}:
            fallback += "\n\n调控目标：" + control["objective"] + "。\n" + "\n".join(f"- {item['station_id']}：{item['action']}" for item in control["actions"])
        return AssistantResult(_humanize_answer(fallback), source_rows, steps, {**live, "control_plan": control} if live_needed else live, "live_fallback" if live_needed else "general", "medium" if useful_hits else "low")
    context = "\n".join(f"[{hit.document.source_id}] {hit.document.text}" for hit in useful_hits)
    if disco_result is not None:
        context += (
            "\n\n[Disco-RAG 知识结论]\n" + disco_result.answer
            + "\n\n[块内 RST 结构]\n" + "\n".join(tree.serialize() for tree in disco_result.trees)
            + "\n\n[跨块修辞图]\n" + disco_result.graph.serialize()
            + "\n\n[话语规划]\n" + disco_result.plan.serialize()
        )
    web_context = "\n".join(f"[{item['title']}] {item['snippet']} 来源：{item['url']}" for item in web_results)
    history_rows = []
    for item in (history or [])[-8:]:
        role = "用户" if item.get("role") == "user" else "调控智能体"
        content = str(item.get("content") or "").strip()[:1500]
        if content:
            history_rows.append(f"{role}：{content}")
    history_context = "\n".join(history_rows)
    live_text = str(live)
    style = {
        "report": "如果用户语义明确要求正式报告，再输出现状判断、证据、建议动作、约束条件和验证指标五个章节；否则不要套用报告模板。",
        "recommendation": "如果用户是在寻求动作建议，先给结论，再给不超过 5 条带优先级的动作和安全边界；不要为了完整而写成长报告。",
        "explanation": "如果用户是在追问原因，先直接解释为什么，再列出最多 4 条证据和需要核查的变量。",
        "query": "如果用户是在查询数值或状态，先用一两句话直接回答，再补充必要的时间戳和数据来源。",
        "general": "按自然对话回答，围绕用户问题给出最短充分答案；只有用户明确要求时才写报告。",
    }[intent]
    prompt = f"最近对话上下文：{history_context or '无'}\n当前问题：{question}\n规则引擎给出的初步意图（仅在语义不明确或模型不可用时兜底）：{intent}\n调控目标参考：{_control_objective(question)}\n运行数据上下文（必须优先使用）：{live_text}\n其中 recent_power_4h 是最近4小时15分钟历史功率，recent_weather_4h 是测风塔/辐照度历史观测，ultra_short_power_forecast 是历史对比加未来4小时功率预测，nwp_forecast_4h 是已完成UTC→UTC+8转换、并插值为15分钟的未来NWP（同时保留原始UTC时间）。\n规则引擎初步调控计划：{control}\n知识库资料：{context or '本次未命中专用知识库'}\n联网公开资料：{web_context or '本次未开启联网搜索或未命中公开资料'}\n请把语义理解放在规则标签之前：结合当前问题、语气、疑问词和最近对话判断用户真正想要查询、解释、建议还是正式报告。若语义与初步意图不一致，以语义为准。实时数据始终以本轮读取结果为准，不要声称未请求实时数据。{style}知识库未命中时允许使用通用电力系统知识，但必须明确标注为通用建议。不要虚构调度指令、设备状态或未提供的历史措施。面向调度员输出自然语言，不要输出 Python 字典、JSON、key=value 字段串或 None/True/False；时间、功率、风速和限电状态请改写成中文标签和单位。联网资料仅作补充，必须保留来源链接，不得用它覆盖实时场站数据或调度规程。最终可见答案控制在 1200 个中文字符以内，优先回答用户当前问题，不要重复上下文。"
    try:
        answer = DeepSeekClient(settings).generate([{"role": "system", "content": "你是新能源消纳与并网智能调控平台的报告智能体。"}, {"role": "user", "content": prompt}])
    except Exception as exc:
        answer = ""
        steps.append({"agent": "报告生成智能体", "status": "fallback", "detail": f"模型暂不可用，已切换实时规则兜底：{type(exc).__name__}"})
    if not answer.strip():
        # Reasoning models can occasionally exhaust the completion budget while
        # returning an empty visible content field. Never expose a blank answer.
        answer = _live_fallback(question, live) if live_needed else _general_fallback(question, bool(useful_hits))
        if live_needed and control and intent in {"report", "recommendation", "explanation"}:
            answer += "\n\n调控目标：" + control["objective"] + "。\n" + "\n".join(f"- {item['station_id']}：{item['action']}" for item in control["actions"])
        steps.append({"agent": "报告生成智能体", "status": "fallback", "detail": "模型未返回可见文本，已使用实时数据与规则引擎生成答案"})
        return AssistantResult(_humanize_answer(answer), source_rows, steps, {**live, "control_plan": control} if live_needed else live, "live_fallback" if live_needed else "general", "medium" if useful_hits else "low")
    steps.append({"agent": "报告生成智能体", "status": "done", "detail": "已结合实时数据与知识库生成报告"})
    return AssistantResult(_humanize_answer(answer), source_rows, steps, {**live, "control_plan": control} if live_needed else live, "rag_live" if live_needed and useful_hits else ("general" if not useful_hits else "rag"), "high" if useful_hits else "medium")
