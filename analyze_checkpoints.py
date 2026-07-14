import argparse
import json
from pathlib import Path
from collections import Counter
from email_notifier import load_smtp_env_from_app_password, send_raw_email, is_configured


def analyze_dir(base: Path, do_email: bool = False):
    any_found = False
    report_lines = []
    for path in sorted(base.glob('checkpoint_*.json')):
        any_found = True
        with open(path, 'r') as f:
            state = json.load(f)
        flagged = state.get('flagged_bar_epochs', [])
        processed = state.get('processed_bar_epochs', [])
        trades = state.get('trades', [])
        label = path.name[len('checkpoint_'):-len('.json')]
        report_lines.append(f'FILE: {path.name}')
        report_lines.append(f'  flagged bars: {len(flagged)}')
        report_lines.append(f'  processed bars: {len(processed)}')
        report_lines.append(f'  progress: {round(100 * len(processed) / len(flagged), 2) if flagged else None}%')
        skipped = state.get('skipped_bar_epochs', [])
        dropped = state.get('dropped_bar_epochs', [])
        report_lines.append(f'  trades saved: {len(trades)}')
        report_lines.append(f'  skipped bars (incomplete): {len(skipped)}')
        report_lines.append(f'  dropped bars (no entry): {len(dropped)}')
        print(f'FILE: {path.name}')
        print(f'  flagged bars: {len(flagged)}')
        print(f'  processed bars: {len(processed)}')
        print(f'  progress: {round(100 * len(processed) / len(flagged), 2) if flagged else None}%')
        print(f'  trades saved: {len(trades)}')
        print(f'  skipped bars (incomplete): {len(skipped)}')
        print(f'  dropped bars (no entry): {len(dropped)}')
        if trades:
            combo_counts = Counter(t.get('combo') for t in trades)
            cat_counts = Counter()
            outcome_counts = Counter()
            win_loss = Counter()
            for t in trades:
                for cat in t.get('categories', []):
                    cat_counts[cat] += 1
                for dur, win in t.get('outcomes', {}).items():
                    dur_int = int(dur)
                    outcome_counts[dur_int] += 1
                    win_loss[(dur_int, bool(win))] += 1
            report_lines.append(f'  combo counts: {dict(combo_counts)}')
            report_lines.append(f'  category counts: {dict(cat_counts)}')
            report_lines.append(f'  durations with outcomes: {dict(sorted(outcome_counts.items()))}')
            print('  combo counts:', dict(combo_counts))
            print('  category counts:', dict(cat_counts))
            print('  durations with outcomes:', dict(sorted(outcome_counts.items())))
            for dur in sorted({d for d, _ in win_loss}):
                wins = win_loss[(dur, True)]
                losses = win_loss[(dur, False)]
                total = wins + losses
                rate = round(100 * wins / total, 2) if total else None
                report_lines.append(f'    duration {dur} min: {wins}W/{losses}L ({rate}%)')
                print(f'    duration {dur} min: {wins}W/{losses}L ({rate}%)')
        report_lines.append('')
        print()

    if do_email:
        load_smtp_env_from_app_password()
        if is_configured():
            subject = '[Backtest] analyze checkpoints summary'
            body = '\n'.join(report_lines)
            ok = send_raw_email(subject, body)
            print(f'email sent: {ok}')
        else:
            print('email skipped: SMTP not configured')

    if not any_found:
        print(f'No checkpoint files found in {base!s}')

    for path in sorted(base.glob('candles_*.csv')):
        print(f'CANDLE CACHE: {path.name} size={path.stat().st_size} bytes')


def main():
    parser = argparse.ArgumentParser(description='Analyze backtest checkpoint files')
    parser.add_argument('--dir', default='backtest_checkpoints', help='Directory containing checkpoints')
    parser.add_argument('--email', action='store_true', help='Send summary email for each checkpoint (requires SMTP env vars)')
    args = parser.parse_args()

    base = Path(args.dir)
    analyze_dir(base, do_email=args.email)


if __name__ == '__main__':
    main()
