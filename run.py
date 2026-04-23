"""
run.py — Ez-Money-Glitch CLI entrypoint.

Usage:
    python run.py train              Train the neural network
    python run.py backtest           Backtest all strategies
    python run.py research           Run auto-researcher (continuous)
    python run.py signals AAPL       Show signals for a ticker
    python run.py signals --mode swing   Scan all swing candidates
    python run.py dashboard          Start dashboard server on localhost:8080

Options:
    python run.py train --sp500 --years 13 --epochs 80
    python run.py backtest --tickers AAPL MSFT NVDA --years 5
    python run.py research --hours 8
    python run.py signals AAPL NVDA --no-nn
"""

import sys
import argparse
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))


def cmd_train(args):
    from nn.train import train
    from data.universe import get_swing_candidates, get_sp500

    if args.tickers:
        tickers = args.tickers
    elif getattr(args, 'sp500', False):
        tickers = get_sp500()
        print(f"Training on full S&P 500: {len(tickers)} tickers")
    else:
        tickers = get_swing_candidates()
        print(f"Training on {len(tickers)} swing candidates")

    train(
        tickers=tickers,
        years=getattr(args, 'years', 13),
        epochs=getattr(args, 'epochs', 80),
        weight_decay=getattr(args, 'weight_decay', 3e-3),
        batch_size=getattr(args, 'batch_size', 128),
    )


def cmd_backtest(args):
    from strategies.backtest import run_all_strategies
    from data.universe import get_swing_candidates

    tickers = getattr(args, 'tickers', None) or get_swing_candidates()[:20]
    years   = getattr(args, 'years', 13)
    print(f"Backtesting {len(tickers)} tickers over {years} years…")
    run_all_strategies(tickers=tickers, years=years)


def cmd_research(args):
    from researcher.auto_researcher import run_researcher
    from data.universe import get_swing_candidates

    tickers  = getattr(args, 'tickers', None)
    hours    = getattr(args, 'hours', None)
    dry_run  = getattr(args, 'dry_run', False)
    years    = getattr(args, 'years', 13)

    run_researcher(tickers=tickers, years=years, max_hours=hours, dry_run=dry_run)


def cmd_signals(args):
    from nn.signals import scan_universe, scan_ticker, print_report
    from data.universe import get_swing_candidates, get_sp500

    no_nn = getattr(args, 'no_nn', False)

    if getattr(args, 'tickers', None):
        tickers = args.tickers
    elif getattr(args, 'mode', 'swing') == 'sp500':
        tickers = get_sp500()[:50]
    else:
        tickers = get_swing_candidates()

    results = scan_universe(tickers, use_nn=not no_nn)
    print_report(results)

    output = getattr(args, 'output', None)
    if output:
        import json
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to: {output}")


def cmd_dashboard(args):
    import os
    import http.server
    import socketserver

    dashboard_dir = _PROJECT_ROOT / "dashboard"
    os.chdir(dashboard_dir)
    port = getattr(args, 'port', 8080)

    with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"Dashboard: http://localhost:{port}/index.html")
        print("Press Ctrl+C to stop.")
        httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(
        description="Ez-Money-Glitch trading research CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train the neural network")
    p_train.add_argument("--tickers", nargs="+")
    p_train.add_argument("--sp500", action="store_true", help="Train on full S&P 500")
    p_train.add_argument("--years", type=int, default=13)
    p_train.add_argument("--epochs", type=int, default=80)
    p_train.add_argument("--weight-decay", type=float, default=3e-3, dest="weight_decay")
    p_train.add_argument("--batch-size", type=int, default=128, dest="batch_size")

    # backtest
    p_bt = sub.add_parser("backtest", help="Backtest all strategies")
    p_bt.add_argument("--tickers", nargs="+")
    p_bt.add_argument("--years", type=int, default=13)

    # research
    p_res = sub.add_parser("research", help="Run auto-researcher")
    p_res.add_argument("--tickers", nargs="+")
    p_res.add_argument("--years", type=int, default=13)
    p_res.add_argument("--hours", type=float, default=None)
    p_res.add_argument("--dry-run", action="store_true", dest="dry_run")

    # signals
    p_sig = sub.add_parser("signals", help="Generate signals for tickers")
    p_sig.add_argument("tickers", nargs="*")
    p_sig.add_argument("--mode", choices=["swing", "sp500"], default="swing")
    p_sig.add_argument("--no-nn", action="store_true", dest="no_nn")
    p_sig.add_argument("--output", default=None, help="Save results to JSON file")

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Start dashboard server")
    p_dash.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()

    commands = {
        "train":     cmd_train,
        "backtest":  cmd_backtest,
        "research":  cmd_research,
        "signals":   cmd_signals,
        "dashboard": cmd_dashboard,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
