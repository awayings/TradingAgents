"""Headless batch runner over a themed watchlist.

``run_analysis.py`` analyses one ticker per invocation and the interactive CLI
is a wizard; neither is usable for a hundred-name overnight sweep. This script
pins the whole sweep to one command:

    .venv/bin/python run_batch_analysis.py                       # 跑清单里的全部
    .venv/bin/python run_batch_analysis.py --themes CPO,PCB      # 只跑部分主题
    .venv/bin/python run_batch_analysis.py --limit 3 --dry-run   # 先干跑看看

The watchlist (``--watchlist``, default ``watchlists/theme_pool_2026-09.json``)
maps a theme to tiers of ``{"code", "name"}`` entries. The pipeline is
company-scoped — a theme is not an input to any agent — so a name that appears
in several themes is analysed once and only the *ordering* and the summary
grouping follow the watchlist. Output is self-contained under ``--out-dir``
(default ``reports/batch_<trade_date>``)::

    reports/batch_2026-09-21/
    ├── reports/300408.SZ_三环集团/complete_report.md   # 与 CLI 相同的报告树
    ├── 300408.SZ/TradingAgentsStrategy_logs/*.json     # propagate 落的全量 state
    ├── batch.log            # 运行日志
    ├── batch_state.json     # 逐只状态，可断点续跑
    └── summary.csv / summary.md   # 汇总

Everything else — provider, model, language, debate depth, vendors — comes from
DEFAULT_CONFIG plus the ``TRADINGAGENTS_*`` overrides in ``.env``, exactly as
the CLI and ``run_analysis.py`` apply them.

Concurrency: ``--workers`` runs that many tickers at once, each thread owning one
``TradingAgentsGraph`` (the graph mutates per-run state such as ``self.ticker``,
so instances cannot be shared). The TDX data layer serializes its own socket
access internally, so parallel workers only contend on the LLM endpoint.

Re-running the same command resumes: an already-completed ticker is skipped
unless ``--force`` is given. Ctrl-C stops scheduling new tickers, lets the
in-flight ones finish, and still writes the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

# Load .env from the repo root, not the working directory, so a scheduled or
# nohup'd run behaves the same from anywhere. This must happen before
# ``tradingagents`` is imported: its __init__ builds DEFAULT_CONFIG at import
# time by reading these variables.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_ROOT / ".env", override=False)
load_dotenv(_REPO_ROOT / ".env.enterprise", override=False)

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402

logger = logging.getLogger("batch")

_DEFAULT_WATCHLIST = _REPO_ROOT / "watchlists" / "theme_pool_2026-09.json"
_DEFAULT_ANALYSTS = ("market", "social", "news", "fundamentals")

# 龙头 sorts ahead of everything else, then the tiers keep the listed order.
_TIER_ORDER = {"龙头": 0, "中军": 1}
# Strength order for the summary table; REVIEW means the decision had no
# parseable rating (#1170) and sorts last among successful runs.
_SIGNAL_ORDER = ["Buy", "Overweight", "Hold", "Underweight", "Sell", "REVIEW"]

_stop = threading.Event()
_state_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Watchlist
# --------------------------------------------------------------------------- #


@dataclass
class Stock:
    """One ticker plus every watchlist slot it occupies."""

    code: str
    name: str
    themes: list[str] = field(default_factory=list)
    tiers: list[str] = field(default_factory=list)
    first_seen: int = 0

    @property
    def tier(self) -> str:
        """The strongest tier this name is listed under (龙头 beats 中军)."""
        ranked = sorted(self.tiers, key=lambda t: _TIER_ORDER.get(t, 99))
        return ranked[0] if ranked else ""

    @property
    def sort_key(self) -> tuple[int, int]:
        return (_TIER_ORDER.get(self.tier, 99), self.first_seen)


def load_watchlist(path: Path) -> list[Stock]:
    """Flatten a themed watchlist into unique tickers, preserving the listed order."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_code: dict[str, Stock] = {}
    for theme in payload.get("themes", []):
        theme_name = str(theme.get("theme") or "").strip()
        for tier, entries in (theme.get("tiers") or {}).items():
            for entry in entries:
                code = str(entry["code"]).strip().upper()
                stock = by_code.get(code)
                if stock is None:
                    stock = Stock(
                        code=code,
                        name=str(entry.get("name") or "").strip(),
                        first_seen=len(by_code),
                    )
                    by_code[code] = stock
                if theme_name and theme_name not in stock.themes:
                    stock.themes.append(theme_name)
                if tier and tier not in stock.tiers:
                    stock.tiers.append(tier)
    return list(by_code.values())


def select(stocks: list[Stock], args) -> list[Stock]:
    """Apply the CLI's theme/tier/only/exclude/limit filters, then order."""
    picked = stocks
    if args.themes:
        wanted = {t.strip() for t in args.themes.split(",") if t.strip()}
        picked = [s for s in picked if wanted & set(s.themes)]
    if args.tiers:
        wanted = {t.strip() for t in args.tiers.split(",") if t.strip()}
        picked = [s for s in picked if wanted & set(s.tiers)]
    if args.only:
        # 代码或中文名都能筛；名字按子串匹配，方便只跑「三环」这种局部。
        tokens = [t.strip() for t in args.only.split(",") if t.strip()]
        picked = [
            s
            for s in picked
            if s.code in tokens or s.name in tokens or any(tok in s.name for tok in tokens)
        ]
    if args.exclude:
        tokens = {t.strip() for t in args.exclude.split(",") if t.strip()}
        picked = [s for s in picked if s.code not in tokens and s.name not in tokens]
    if args.order == "tier":
        picked = sorted(picked, key=lambda s: s.sort_key)
    if args.limit:
        picked = picked[: args.limit]
    return picked


# --------------------------------------------------------------------------- #
# Pre-flight validation
# --------------------------------------------------------------------------- #


def validate(stocks: list[Stock]) -> tuple[list[Stock], list[dict]]:
    """Check every code against the quote server and drop the ones that fail.

    Catches the two ways a hand-maintained watchlist rots: a typo'd code, and a
    code that resolves but belongs to a different company (renames, wrong board)
    — which would otherwise silently produce a confident report on the wrong
    stock. A name mismatch is corrected in place rather than fatal, since the
    code is what the pipeline actually queries.
    """
    from tradingagents.dataflows import tdx_common

    ok: list[Stock] = []
    bad: list[dict] = []
    for stock in stocks:
        try:
            snapshot = tdx_common.quote_snapshot(stock.code)
        except Exception as exc:  # noqa: BLE001 - a bad symbol must not kill the sweep
            bad.append({"code": stock.code, "name": stock.name, "reason": str(exc)})
            continue
        actual = str(snapshot.get("name") or "").strip()
        if not actual:
            bad.append({"code": stock.code, "name": stock.name, "reason": "行情源未返回该代码"})
            continue
        if actual != stock.name:
            logger.warning(
                "%s 清单名为「%s」，行情源返回「%s」，按行情源名继续",
                stock.code,
                stock.name,
                actual,
            )
            stock.name = actual
        ok.append(stock)
    return ok, bad


# --------------------------------------------------------------------------- #
# Output layout
# --------------------------------------------------------------------------- #


def _safe_dir_name(name: str) -> str:
    """A filename-safe company name: no separators, no traversal."""
    return re.sub(r'[\\/:*?"<>|\s]+', "", name) or "unnamed"


def report_dir(out_dir: Path, stock: Stock) -> Path:
    return out_dir / "reports" / f"{stock.code}_{_safe_dir_name(stock.name)}"


def completed_report(out_dir: Path, stock: Stock) -> Path | None:
    """The existing complete_report.md for ``stock``, or None.

    Globs as well as checking the canonical directory so a ticker renamed
    between runs still resumes instead of re-running.
    """
    canonical = report_dir(out_dir, stock) / "complete_report.md"
    if canonical.exists():
        return canonical
    for candidate in (out_dir / "reports").glob(f"{stock.code}_*/complete_report.md"):
        return candidate
    return None


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("%s 解析失败，当作全新运行处理", path)
    return {}


def save_state(path: Path, state: dict) -> None:
    """Write the state file atomically so a kill mid-write cannot corrupt it."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

_thread_local = threading.local()


def _graph_for(analysts: tuple[str, ...], config: dict) -> TradingAgentsGraph:
    """One graph per worker thread, reused across that thread's tickers.

    ``propagate`` mutates per-run state on the instance (``self.ticker``,
    ``self.curr_state``, the state log), so a single instance cannot serve two
    concurrent runs; building one per thread keeps the cost to ``--workers``
    constructions for the whole sweep.
    """
    graph = getattr(_thread_local, "graph", None)
    if graph is None:
        graph = TradingAgentsGraph(list(analysts), config=config, debug=False)
        _thread_local.graph = graph
    return graph


def analyze(
    stock: Stock,
    trade_date: str,
    analysts: tuple[str, ...],
    config: dict,
    out_dir: Path,
    retries: int,
) -> dict:
    """Run the full graph for one ticker and write its report tree."""
    started = time.time()
    save_path = report_dir(out_dir, stock)
    last_error = ""
    for attempt in range(retries + 1):
        if attempt:
            logger.warning("%s (%s) 第 %d 次重试", stock.code, stock.name, attempt)
            time.sleep(10 * attempt)
        try:
            graph = _graph_for(analysts, config)
            final_state, signal = graph.propagate(stock.code, trade_date)
            graph.save_reports(final_state, stock.code, save_path=save_path)
            return {
                "code": stock.code,
                "name": stock.name,
                "tier": stock.tier,
                "themes": "/".join(stock.themes),
                "status": "done",
                "signal": str(signal),
                "duration_s": round(time.time() - started, 1),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "report": str(save_path / "complete_report.md"),
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not end the sweep
            last_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "%s (%s) 失败: %s",
                stock.code,
                stock.name,
                last_error,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
    return {
        "code": stock.code,
        "name": stock.name,
        "tier": stock.tier,
        "themes": "/".join(stock.themes),
        "status": "failed",
        "signal": "",
        "duration_s": round(time.time() - started, 1),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "report": "",
        "error": last_error,
    }


def _run(
    stock: Stock,
    progress: dict,
    trade_date: str,
    analysts: tuple[str, ...],
    config: dict,
    out_dir: Path,
    retries: int,
) -> dict:
    """Run one ticker, recording when the pool actually starts it.

    ``ThreadPoolExecutor`` queues everything handed to it, so the submit loop
    cannot tell a running ticker from a backlogged one — only the worker thread
    can. Marking the start here is what keeps the heartbeat's in-flight line
    and its ETA honest on a 100-name sweep.
    """
    label = f"{stock.code} {stock.name}"
    with _state_lock:
        progress["running"][label] = time.time()
    try:
        return analyze(stock, trade_date, analysts, config, out_dir, retries)
    finally:
        with _state_lock:
            progress["running"].pop(label, None)


def _eta_seconds(progress: dict, now: float, workers: int) -> float | None:
    """Estimated wall-clock seconds until the queue drains, or None if unknown.

    Built from the measured per-ticker duration divided by the worker count,
    not from ``finished / elapsed``: each ticker occupies one worker, so
    wall-clock ≈ thread-time / workers, and a completion rate reads several
    times too slow through the whole first wave — the window where every
    worker is still busy with its first ticker and nothing has finished yet,
    which is exactly when a 100-name sweep says "预计剩余 594.0 分钟".
    """
    if not progress["durations"]:
        return None
    mean = sum(progress["durations"]) / len(progress["durations"])
    finished = progress["completed"] + progress["failed"]
    not_started = max(progress["pending"] - finished - len(progress["running"]), 0)
    remaining = not_started * mean
    # A ticker already in flight owes only the rest of its own slot.
    for started_at in progress["running"].values():
        remaining += max(mean - (now - started_at), 0)
    return remaining / max(workers, 1)


def _heartbeat(progress: dict, total: int, workers: int, started: float) -> None:
    """Log a periodic ETA line so an unattended run is legible in the morning."""
    while not _stop.wait(120):
        now = time.time()
        with _state_lock:
            completed = progress["completed"]
            failed = progress["failed"]
            skipped = progress["skipped"]
            running = ", ".join(progress["running"]) or "-"
            running_count = len(progress["running"])
            eta = _eta_seconds(progress, now, workers)
        logger.info(
            "进度 %d/%d（完成 %d 失败 %d 跳过 %d），在跑 %d 只: %s，已耗时 %.1f 分钟，预计剩余 %s",
            completed + failed + skipped,
            total,
            completed,
            failed,
            skipped,
            running_count,
            running,
            (now - started) / 60,
            f"{eta / 60:.1f} 分钟" if eta is not None else "估算中（还没有完成的样本）",
        )


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def write_summary(
    out_dir: Path, results: list[dict], invalid: list[dict], trade_date: str, elapsed_s: float
) -> None:
    """Write summary.csv and summary.md grouped by theme."""
    order = {code: i for i, code in enumerate(_SIGNAL_ORDER)}

    def rank(row: dict) -> int:
        """Strongest rating first; failures sink to the bottom."""
        if row["status"] != "done":
            return len(_SIGNAL_ORDER) + 1
        return order.get(row["signal"], len(_SIGNAL_ORDER))

    fields = [
        "code",
        "name",
        "tier",
        "themes",
        "status",
        "signal",
        "duration_s",
        "finished_at",
        "report",
        "error",
    ]
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in sorted(results, key=lambda r: (r["tier"], r["code"])):
            writer.writerow({k: row.get(k, "") for k in fields})

    counts: dict[str, int] = {}
    failed = 0
    for row in results:
        if row["status"] == "done":
            counts[row["signal"]] = counts.get(row["signal"], 0) + 1
        else:
            failed += 1

    lines = [
        f"# 主题池批量分析汇总 · {trade_date}",
        "",
        f"- 标的数：{len(results)}（完成 {len(results) - failed}，失败 {failed}）",
        "- 评级分布："
        + "、".join(
            f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: order.get(kv[0], 99))
        ),
        f"- 耗时：{elapsed_s / 60:.1f} 分钟",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
    ]
    if invalid:
        lines += ["", "## 未通过前置校验（未分析）", ""]
        lines += [f"- `{r['code']}` {r['name']}：{r['reason']}" for r in invalid]

    by_theme: dict[str, list[dict]] = {}
    for row in results:
        for theme in (row["themes"] or "未分组").split("/"):
            by_theme.setdefault(theme, []).append(row)

    for theme in sorted(by_theme):
        lines += [
            "",
            f"## {theme}",
            "",
            "| 层级 | 代码 | 名称 | 评级 | 耗时(分) | 报告 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in sorted(by_theme[theme], key=rank):
            label = row["signal"] if row["status"] == "done" else row["status"]
            if row["status"] == "done" and row["report"]:
                rel = Path(row["report"]).relative_to(out_dir)
                report_cell = f"[报告]({rel.as_posix()})"
            else:
                report_cell = row["error"][:60] or "-"
            lines.append(
                f"| {row['tier'] or '-'} | {row['code']} | {row['name']} | "
                f"{label} | {row['duration_s'] / 60:.1f} | {report_cell} |"
            )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按主题池清单批量运行 TradingAgents 分析（可断点续跑）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--watchlist", default=str(_DEFAULT_WATCHLIST), help="主题池 JSON 路径")
    parser.add_argument(
        "--date", default=datetime.now().strftime("%Y-%m-%d"), help="分析日期 YYYY-MM-DD，默认今天"
    )
    parser.add_argument("--out-dir", default=None, help="输出根目录，默认 reports/batch_<日期>")
    parser.add_argument(
        "--workers", type=int, default=3, help="并发分析数，默认 3（每个线程一个 graph 实例）"
    )
    parser.add_argument(
        "--analysts",
        default=",".join(_DEFAULT_ANALYSTS),
        help="分析师组合，默认 market,social,news,fundamentals",
    )
    parser.add_argument(
        "--order",
        choices=("tier", "list"),
        default="tier",
        help="tier=龙头优先（默认），list=严格按清单顺序",
    )
    parser.add_argument("--themes", default=None, help="只跑这些主题，逗号分隔")
    parser.add_argument("--tiers", default=None, help="只跑这些层级，如 龙头")
    parser.add_argument("--only", default=None, help="只跑这些代码或名称（支持名字子串）")
    parser.add_argument("--exclude", default=None, help="排除这些代码或名称")
    parser.add_argument("--limit", type=int, default=None, help="最多分析多少只（先小样本试跑）")
    parser.add_argument("--retries", type=int, default=1, help="单只失败后的重试次数，默认 1")
    parser.add_argument("--force", action="store_true", help="忽略已完成记录，全部重跑")
    parser.add_argument("--no-preflight", action="store_true", help="跳过代码/名称校验")
    parser.add_argument("--validate-only", action="store_true", help="只校验清单并打印计划")
    parser.add_argument("--dry-run", action="store_true", help="校验清单 + 打印计划，不调用模型")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    trade_date = args.date
    out_dir = Path(args.out_dir) if args.out_dir else _REPO_ROOT / "reports" / f"batch_{trade_date}"
    out_dir = out_dir.resolve()

    # ---- logging ---------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(out_dir / "batch.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # One INFO line per LLM request would bury the batch's own progress lines
    # under thousands of them over a 100-name sweep.
    for noisy in ("httpx", "httpcore", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ---- plan ------------------------------------------------------------
    watchlist_path = Path(args.watchlist)
    if not watchlist_path.exists():
        logger.error("清单不存在: %s", watchlist_path)
        return 2
    all_stocks = load_watchlist(watchlist_path)
    stocks = select(all_stocks, args)
    analysts = tuple(a.strip() for a in args.analysts.split(",") if a.strip())

    logger.info(
        "清单 %s：%d 只（去重后）→ 本次计划 %d 只",
        watchlist_path.name,
        len(all_stocks),
        len(stocks),
    )
    logger.info("分析日期 %s，分析师 %s，并发 %d", trade_date, ",".join(analysts), args.workers)
    if not stocks:
        logger.warning("筛选后没有标的，退出。")
        return 2

    # ---- pre-flight ------------------------------------------------------
    invalid: list[dict] = []
    if not args.no_preflight:
        logger.info("校验 %d 个代码……", len(stocks))
        stocks, invalid = validate(stocks)
        for row in invalid:
            logger.error("校验未通过，跳过: %s %s (%s)", row["code"], row["name"], row["reason"])
        if not stocks:
            logger.error("所有标的都未通过校验，行情源可能不可用。")
            return 2
        logger.info("校验通过 %d 只，未通过 %d 只", len(stocks), len(invalid))

    if args.dry_run or args.validate_only:
        for i, s in enumerate(stocks, 1):
            logger.info(
                "[%d/%d] %s %s | %s | %s",
                i,
                len(stocks),
                s.code,
                s.name,
                s.tier or "-",
                "/".join(s.themes),
            )
        logger.info(
            "%s结束，未调用模型。输出目录将是 %s", "校验" if args.validate_only else "干跑", out_dir
        )
        return 0

    # ---- config ----------------------------------------------------------
    config = DEFAULT_CONFIG.copy()
    # results_dir drives propagate's full-state JSON; pointing it at the batch
    # root keeps one sweep self-contained (the shared OHLCV cache in
    # data_cache_dir and the cross-run memory log stay where they are).
    config["results_dir"] = str(out_dir)
    os.makedirs(config["data_cache_dir"], exist_ok=True)

    state_path = out_dir / "batch_state.json"
    state = load_state(state_path)
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "watchlist": str(watchlist_path),
                "trade_date": trade_date,
                "analysts": list(analysts),
                "workers": args.workers,
                "order": args.order,
                "llm_provider": config.get("llm_provider"),
                "deep_think_llm": config.get("deep_think_llm"),
                "quick_think_llm": config.get("quick_think_llm"),
                "output_language": config.get("output_language"),
                "started_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def handle_sigint(signum, frame):  # noqa: ARG001
        if _stop.is_set():
            logger.warning("再次收到中断信号，立即退出。")
            os._exit(130)
        logger.warning("收到中断信号：不再派发新标的，等待在跑的完成（再次 Ctrl-C 立即退出）。")
        _stop.set()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    # ---- run -------------------------------------------------------------
    pending: list[Stock] = []
    results: list[dict] = []
    skipped = 0
    for stock in stocks:
        row = state.get(stock.code)
        if (
            not args.force
            and row
            and row.get("status") == "done"
            and completed_report(out_dir, stock)
        ):
            results.append(row)
            skipped += 1
            logger.info("跳过（已完成）: %s %s", stock.code, stock.name)
        else:
            pending.append(stock)

    progress = {
        "completed": 0,
        "failed": 0,
        "skipped": skipped,
        "pending": len(pending),
        # Populated by the worker threads (_run), not by the submit loop:
        # only a running thread knows its ticker has actually started.
        "running": {},
        # Per-ticker wall time of the finished ones, the basis for the ETA.
        "durations": [],
    }
    started = time.time()
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=(progress, len(pending) + skipped, args.workers, started),
        daemon=True,
    )
    heartbeat.start()

    logger.info("开始分析 %d 只（跳过 %d 只已完成）", len(pending), skipped)
    failures: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="batch") as pool:
        futures = {}
        for stock in pending:
            if _stop.is_set():
                logger.warning("已中断，剩余 %d 只未派发。", len(pending) - len(futures))
                break
            futures[
                pool.submit(
                    _run, stock, progress, trade_date, analysts, config, out_dir, args.retries
                )
            ] = stock

        # as_completed so each finished ticker is logged and persisted without
        # waiting for the slower ones submitted ahead of it.
        for future in as_completed(futures):
            stock = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - defensive: analyze() already catches
                row = {
                    "code": stock.code,
                    "name": stock.name,
                    "tier": stock.tier,
                    "themes": "/".join(stock.themes),
                    "status": "failed",
                    "signal": "",
                    "duration_s": 0.0,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "report": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            with _state_lock:
                if row["status"] == "done":
                    progress["completed"] += 1
                    results.append(row)
                    progress["durations"].append(row["duration_s"])
                else:
                    progress["failed"] += 1
                    failures.append(row)
                state[row["code"]] = row
                save_state(state_path, state)
            if row["status"] == "done":
                logger.info(
                    "完成 %s %s → %s（%.1f 分钟）",
                    row["code"],
                    row["name"],
                    row["signal"],
                    row["duration_s"] / 60,
                )
            else:
                logger.error("失败 %s %s：%s", row["code"], row["name"], row["error"])

    elapsed = time.time() - started
    _stop.set()
    write_summary(out_dir, results + failures, invalid, trade_date, elapsed)

    logger.info(
        "全部结束：完成 %d，失败 %d，跳过 %d，耗时 %.1f 分钟",
        len(results),
        len(failures),
        skipped,
        elapsed / 60,
    )
    logger.info("汇总: %s", out_dir / "summary.md")
    if failures:
        logger.error("失败清单: %s", "、".join(f"{r['code']} {r['name']}" for r in failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
