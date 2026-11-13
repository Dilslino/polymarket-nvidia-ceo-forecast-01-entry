"""Validate main.py outputs: format, size, determinism, runtime per case.
Usage: python3 tests/validate.py
Expects out/ (run 1) and out2/ (run 2) to exist from the main.py runs.
"""
import json, os, sys, glob, time, subprocess, hashlib

V2 = '/home/hatch/workspace/makefun/workspaces/polymarket-nvidia-ceo-forecast-01-v2'
KIT = '/home/hatch/workspace/makefun/workspaces/polymarket-nvidia-ceo-forecast-01-kit'
CASES = ['p00', 'p01', 'p02', 'p03', 'p04', 'p05']

def check_outdir(d):
    for cid in CASES:
        p = os.path.join(d, f'{cid}.out')
        assert os.path.exists(p), f'missing {p}'
        size = os.path.getsize(p)
        assert size <= 1048576, f'{cid} too large: {size}'
        out = json.load(open(p))
        assert set(out.keys()) == {'forecasts'}, out.keys()
        case = json.load(open(f'{KIT}/input/{cid}.in'))
        assert len(out['forecasts']) == len(case['markets']), cid
        ids = {f['id'] for f in out['forecasts']}
        assert ids == {m['id'] for m in case['markets']}, f'{cid} id mismatch'
        for f in out['forecasts']:
            assert len(f['q']) == 240, (cid, f['id'])
            for row in f['q']:
                assert len(row) == 5
                assert all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in row)
                assert all(row[j] <= row[j + 1] for j in range(4)), f'{cid} not monotone'
        print(f'{cid}: OK ({size} bytes)')

def sha_dir(d):
    h = hashlib.sha256()
    for cid in CASES:
        h.update(open(os.path.join(d, f'{cid}.out'), 'rb').read())
    return h.hexdigest()

if __name__ == '__main__':
    print('--- format check (out/) ---')
    check_outdir(f'{V2}/out')
    print('--- determinism (out vs out2) ---')
    s1 = sha_dir(f'{V2}/out')
    s2 = sha_dir(f'{V2}/out2')
    print('sha256 out :', s1)
    print('sha256 out2:', s2)
    assert s1 == s2, 'NOT deterministic!'
    print('BYTE-IDENTICAL across 2 runs: OK')
    print('--- runtime per case (single-case runs) ---')
    for cid in CASES:
        # run main.py on a single-case input dir
        import tempfile, shutil
        tmp_in = tempfile.mkdtemp()
        tmp_out = tempfile.mkdtemp()
        shutil.copy(f'{KIT}/input/{cid}.in', tmp_in)
        env = dict(os.environ, MAKEFUN_INPUT_DIR=tmp_in, MAKEFUN_OUTPUT_DIR=tmp_out)
        t0 = time.time()
        subprocess.run([sys.executable, f'{V2}/main.py'], env=env,
                       check=True, capture_output=True)
        dt = time.time() - t0
        print(f'{cid}: {dt:.3f}s')
        shutil.rmtree(tmp_in); shutil.rmtree(tmp_out)
