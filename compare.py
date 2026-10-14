#!/usr/bin/env python3
"""Small comparison runner: raw factor milliseconds and geometric speedups."""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parent
VERSIONS = ['glu-original', 'c94', 'c104', 'c106']


def factor_ms(report):
    phases = [p for p in report['diagnostic_phases'] if p['name'] == 'numeric_factorization']
    if len(phases) != 1:
        raise ValueError('Expected exactly one numeric_factorization interval')
    duration = phases[0]['end_ms'] - phases[0]['start_ms']
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid factorization duration')
    return duration


def vector(path, n):
    values = [float(line) for line in path.read_text().splitlines() if line.strip()]
    if len(values) != n or not all(math.isfinite(v) for v in values):
        raise ValueError('Invalid solution vector: ' + str(path))
    return values


def check_solution(path, reference):
    actual = vector(path, len(reference))
    numerator = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(actual, reference)))
    denominator = math.sqrt(math.fsum(b * b for b in reference))
    error = numerator / max(denominator, 1e-300)
    if not math.isfinite(error) or error > 1e-6:
        raise ValueError('Solution relative L2 exceeds 1e-6: ' + str(error))
    return error


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--versions', nargs='+', choices=VERSIONS, default=VERSIONS)
    parser.add_argument('--matrices', nargs='+', help='Optional names, e.g. add32_csr G2_circuit_csr')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1 or args.timeout <= 0:
        parser.error('Invalid warmups, repeats, or timeout')
    if len(set(args.versions)) != len(args.versions) or 'glu-original' not in args.versions:
        parser.error('Use unique versions including glu-original as baseline')
    cases = json.loads((ROOT / 'matrices/manifest.json').read_text())
    if args.matrices:
        if len(set(args.matrices)) != len(args.matrices) or set(args.matrices) - {c['id'] for c in cases}:
            parser.error('Unknown or duplicate matrix name')
        cases = [c for c in cases if c['id'] in args.matrices]
    binaries = {v: ROOT / 'versions' / v / 'src/lu_cmd' for v in args.versions}
    for path in binaries.values():
        if not path.is_file() or not os.access(str(path), os.X_OK):
            parser.error('Build the binary first: ' + str(path))
    for case in cases:
        if sha(ROOT / case['matrix']) != case['matrix_sha256'] or sha(ROOT / case['reference']) != case['reference_sha256']:
            raise ValueError('Dataset checksum mismatch: ' + case['id'])
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    output = ROOT / 'results' / stamp
    output.mkdir(parents=True)
    (output / 'config.json').write_text(json.dumps({**vars(args), 'binary_sha256': {v: sha(p) for v, p in binaries.items()},
                                                   'matrix_names': [c['id'] for c in cases]}, indent=2) + '\n')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    rows, raw = [], []
    print('Results: ' + str(output), flush=True)
    try:
        for case in cases:
            reference = vector(ROOT / case['reference'], case['n'])
            baseline = None
            for version in ['glu-original'] + [v for v in args.versions if v != 'glu-original']:
                samples = []
                for trial in range(args.warmups + args.repeats):
                    folder = output / version / case['id'] / ('trial-%02d' % trial)
                    folder.mkdir(parents=True)
                    start = time.perf_counter()
                    with (folder / 'stdout.log').open('w') as out, (folder / 'stderr.log').open('w') as err:
                        subprocess.run([str(binaries[version]), '-i', str(ROOT / case['matrix'])],
                                       cwd=str(folder), env=env, stdout=out, stderr=err,
                                       timeout=args.timeout, check=True)
                    process_ms = (time.perf_counter() - start) * 1000
                    duration = factor_ms(json.loads((folder / 'timings.json').read_text()))
                    error = check_solution(folder / 'x.dat', reference)
                    item = {'version': version, 'matrix': case['id'], 'trial': trial,
                            'measured': trial >= args.warmups, 'numeric_factorization_ms': duration,
                            'process_ms': process_ms, 'relative_l2': error}
                    raw.append(item)
                    with (output / 'samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(item) + '\n')
                    if item['measured']:
                        samples.append(item)
                mean = statistics.mean(s['numeric_factorization_ms'] for s in samples)
                if version == 'glu-original':
                    baseline = mean
                row = {'version': version, 'matrix': case['id'], 'numeric_factorization_ms': mean,
                       'factor_min_ms': min(s['numeric_factorization_ms'] for s in samples),
                       'factor_max_ms': max(s['numeric_factorization_ms'] for s in samples),
                       'process_ms': statistics.mean(s['process_ms'] for s in samples),
                       'speedup_vs_glu': baseline / mean}
                rows.append(row)
                write_csv(output / 'times.csv', rows)
                print('%-12s %-22s factor=%10.3f ms  process=%10.3f ms  speedup=%.4fx' %
                      (version, case['id'], mean, row['process_ms'], row['speedup_vs_glu']), flush=True)
        summary = []
        for version in args.versions:
            values = [r['speedup_vs_glu'] for r in rows if r['version'] == version]
            gm = math.exp(statistics.mean(math.log(v) for v in values))
            summary.append({'version': version, 'matrix_count': len(values), 'geomean_speedup_vs_glu': gm})
            print('%s: geometric mean speedup %.6fx (%d matrices)' % (version, gm, len(values)))
        write_csv(output / 'summary.csv', summary)
        (output / 'status.json').write_text(json.dumps({'status': 'completed', 'checked_vectors': len(raw)}, indent=2) + '\n')
    except BaseException as error:
        (output / 'status.json').write_text(json.dumps({'status': 'failed', 'error': repr(error)}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
