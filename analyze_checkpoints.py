import json
from pathlib import Path
from collections import Counter

base = Path(r'c:/Users/Arowoshola Shola/Desktop/Trade/backtest_checkpoints')
for path in sorted(base.glob('checkpoint_*.json')):
    with open(path, 'r') as f:
        state = json.load(f)
    flagged = state.get('flagged_bar_epochs', [])
    processed = state.get('processed_bar_epochs', [])
    trades = state.get('trades', [])
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

for path in sorted(base.glob('candles_*.csv')):
    print(f'CANDLE CACHE: {path.name} size={path.stat().st_size} bytes')
