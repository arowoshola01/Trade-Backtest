#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export BACKTEST_SMTP_HOST="smtp.gmail.com"
export BACKTEST_SMTP_PORT="587"
export BACKTEST_SMTP_USERNAME="arowoshola01@gmail.com"
export BACKTEST_SMTP_FROM="arowoshola01@gmail.com"
export BACKTEST_SMTP_TO="arowoshola01@gmail.com"

echo "Gmail SMTP is configured for arowoshola01@gmail.com."
echo "Enter your Gmail app password (not your normal password):"
read -s BACKTEST_SMTP_PASSWORD
export BACKTEST_SMTP_PASSWORD

echo
printf 'Running checkpoint analysis and emailing summaries...\n'
python3 analyze_checkpoints.py --email
