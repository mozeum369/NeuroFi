#!/usr/bin/env python3
import argparse

def main():
    parser = argparse.ArgumentParser(prog="Cryptobot", description="Cryptobot CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("start", help="Start the Cryptobot engine")
    sub.add_parser("dashboard", help="Launch the Cryptobot dashboard")

    args = parser.parse_args()

    if args.cmd == "start":
        from src.bot import run_bot  # Make sure src/bot.py has run_bot()
        run_bot()

    elif args.cmd == "dashboard":
        from src.dashboard import run_dashboard
        run_dashboard()

if __name__ == "__main__":
    main()
