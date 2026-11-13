"""Fetch full Polymarket CLOB 1-min price history for all 24 universe markets
(2026-11-10T02:00Z .. 2026-11-13T06:00Z) and build 1-minute grids.

DEV ONLY — never used by main.py. Used to build extra causal walk-forward
(72h context -> 4h future) splits at checkpoints before p00, so tuning is
not overfit to the 6 practice cases.

Usage: python3 harness/fetch_history.py
Output: harness/history.json  {conditionId: {'name','grid':[price|None]*N,
'grid_start':'2026-11-10T02:00:00.000Z'}}
"""
import json, time, urllib.request, datetime as dt

KIT = '/home/hatch/workspace/makefun/workspaces/polymarket-nvidia-ceo-forecast-01-kit'
case = json.load(open(f'{KIT}/input/p05.in'))
markets = case['markets']
print('markets:', len(markets))

T0 = dt.datetime(2026, 11, 10, 2, 0, 0, tzinfo=dt.timezone.utc)
T1 = dt.datetime(2026, 11, 13, 6, 0, 0, tzinfo=dt.timezone.utc)
start = int(T0.timestamp())
end = int(T1.timestamp())
nmin = int((T1 - T0).total_seconds() // 60) + 1
print('grid minutes:', nmin)

out = {}
for m in markets:
    tok = m['yes_token']
    url = (f'https://clob.polymarket.com/prices-history?market={tok}'
           f'&startTs={start}&endTs={end}&fidelity=1')
    d = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            break
        except Exception as e:
            print('retry', m['name'], e)
            time.sleep(3)
    hist = d.get('history', []) if d else []
    pts = sorted((h['t'], float(h['p'])) for h in hist)
    grid = [None] * nmin
    j = 0
    last = None
    for i in range(nmin):
        mt = int(T0.timestamp()) + i * 60
        while j < len(pts) and pts[j][0] <= mt:
            last = pts[j][1]
            j += 1
        grid[i] = last
    out[m['id']] = {'name': m['name'], 'slug': m['slug'],
                    'yes_token': tok, 'grid': grid,
                    'grid_start': '2026-11-10T02:00:00.000Z',
                    'n_points': len(pts)}
    print(f"{m['name'][:28]:30s} pts={len(pts):5d} nones={grid.count(None)}")

json.dump(out, open('/home/hatch/workspace/makefun/workspaces/polymarket-nvidia-ceo-forecast-01-v2/harness/history.json', 'w'))
print('saved history.json')
