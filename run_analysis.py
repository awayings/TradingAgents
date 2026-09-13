"""Headless fixed-parameter analysis runner.

The interactive CLI (``python -m cli.main``) is a wizard and re-prompts for
ticker, date and analysts on every run. This script pins those parameters on
the command line and reuses the same ``TradingAgentsGraph`` pipeline, so a
repeat analysis is one command:

    .venv/bin/python run_analysis.py 002594.SZ 2026-09-13

Arguments: ``ticker [date] [analysts]`` — date defaults to today, analysts
default to ``market,social,news,fundamentals``. LLM provider, model, output
language, research depth and every other knob come from DEFAULT_CONFIG and the
TRADINGAGENTS_* env-var overrides in ``.env``, exactly as the CLI would apply
them, so this script and the interactive CLI produce the same reports.

Reports land under ``results_dir/<ticker>/<date>/reports`` (the tree layout
from ``write_report_tree``, including the combined ``complete_report.md``), and
the full state JSON under ``results_dir/<ticker>/TradingAgentsStrategy_logs``.
"""

from datetime import datetime
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

_DEFAULT_ANALYSTS = ("market", "social", "news", "fundamentals")


def main(argv: list[str]) -> int:
    ticker = argv[0] if argv else "002594.SZ"
    trade_date = argv[1] if len(argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    analysts = tuple(argv[2].split(",")) if len(argv) > 2 else _DEFAULT_ANALYSTS

    # DEFAULT_CONFIG already applies TRADINGAGENTS_* env-var overrides
    # (llm_provider, deep/quick thinkers, backend_url, language, ...).
    config = DEFAULT_CONFIG.copy()

    print(f"Analyzing {ticker} on {trade_date} with analysts: {', '.join(analysts)}")
    graph = TradingAgentsGraph(analysts, config=config, debug=True)
    final_state, signal = graph.propagate(ticker, trade_date)

    save_path = (
        Path(DEFAULT_CONFIG["results_dir"]) / ticker / trade_date / "reports"
    )
    complete_path = graph.save_reports(final_state, ticker, save_path=save_path)
    print(f"Reports saved to: {save_path}")
    print(f"Combined report: {complete_path}")
    print(f"Signal: {signal}")
    print("Final decision:")
    print(final_state["final_trade_decision"])
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
