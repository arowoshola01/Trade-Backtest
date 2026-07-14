import argparse
import json
from pathlib import Path
from collections import Counter
from email_notifier import send_analysis_summary_email, is_configured


def analyze_dir(base: Path, do_email: bool = False):
    any_found = False
    analysis_items = []
    for path in sorted(base.glob('checkpoint_*.json')):
        any_found = True
        with open(path, 'r') as f:
            state = json.load(f)
        flagged = state.get('flagged_bar_epochs', [])
        processed = state.get('processed_bar_epochs', [])
        trades = state.get('trades', [])
        label = path.name[len('checkpoint_'):-len('.json')]
        print(f'FILE: {path.name}')
        print(f'  flagged bars: {len(flagged)}')
        print(f'  processed bars: {len(processed)}')
        print(f'  progress: {round(100 * len(processed) / len(flagged), 2) if flagged else None}%')
        print(f'  trades saved: {len(trades)}')
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
            print('  combo counts:', dict(combo_counts))
            print('  category counts:', dict(cat_counts))
            print('  durations with outcomes:', dict(sorted(outcome_counts.items())))
            for dur in sorted({d for d, _ in win_loss}):
                wins = win_loss[(dur, True)]
                losses = win_loss[(dur, False)]
                total = wins + losses
                rate = round(100 * wins / total, 2) if total else None
                print(f'    duration {dur} min: {wins}W/{losses}L ({rate}%)')
        print()

        if do_email:
            processed_count = len(processed)
            total_count = len(flagged)
            trades_count = len(trades)
            analysis_items.append({
                'label': label,
                'processed': processed_count,
                'total': total_count,
                'trades': trades_count,
                'progress': round(100 * processed_count / total_count, 2) if total_count else 100.0,
                'combo_counts': dict(Counter(t.get('combo') for t in trades)),
                'category_counts': dict(Counter(cat for t in trades for cat in t.get('categories', []))),
            })

    if do_email:
        if is_configured():
            ok = send_analysis_summary_email(analysis_items)
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
