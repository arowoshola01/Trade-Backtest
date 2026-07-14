#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export BACKTEST_SMTP_HOST="smtp.gmail.com"
export BACKTEST_SMTP_PORT="587"
export BACKTEST_SMTP_USERNAME="arowoshola01@gmail.com"
export BACKTEST_SMTP_FROM="arowoshola01@gmail.com"
export BACKTEST_SMTP_TO="arowoshola01@gmail.com"

echo "Gmail SMTP is configured for arowoshola01@gmail.com."
if [ -f "app password.md" ]; then
  BACKTEST_SMTP_PASSWORD=$(sed -n '1p' "app password.md" | tr -d '\r\n')
  if [ -z "$BACKTEST_SMTP_PASSWORD" ]; then
    echo "app password.md is empty. Please paste your app password on the first line."
    exit 1
  fi
else
  echo "Enter your Gmail app password (not your normal password):"
  read -s BACKTEST_SMTP_PASSWORD
  echo
fi
export BACKTEST_SMTP_PASSWORD

echo
printf 'Starting backtest...\n'
python3 backtest.py
