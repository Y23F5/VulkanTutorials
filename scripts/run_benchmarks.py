#!/usr/bin/env python3
"""
Automated benchmark runner for dissertation experiments.
Enumerates all 513 test configurations and invokes GPUDrivenRendering executable.
"""
import subprocess
import os
import time
from pathlib import Path

EXECUTABLE = "build/GPUDrivenRendering/Debug/GPUDrivenRendering.exe"
OUTPUT_DIR = Path("benchmark_results")

GRID_SIZES  = [16, 32, 64, 128, 256, 512]
CHUNK_SIZES = [4, 8, 16]
DENSITIES   = [20, 50, 80]
SCHEMES     = [1, 2, 3]
SEEDS       = [42, 1337, 9999]

WARMUP_FRAMES = 120
RECORD_FRAMES = 1200


def run_test(grid, chunk, density, scheme, seed, update_size=0):
    output_name = f"g{grid}_c{chunk}_d{density}_s{scheme}_seed{seed}"
    if update_size > 0:
        output_name += f"_u{update_size}"
    output_path = OUTPUT_DIR / f"{output_name}.csv"

    if output_path.exists():
        print(f"  [SKIP] {output_name} -- already exists")
        return True

    cmd = [
        EXECUTABLE, "-Benchmark",
        "-GridSize", str(grid), "-ChunkSize", str(chunk),
        "-Density", str(density), "-Scheme", str(scheme),
        "-Seed", str(seed),
        "-WarmupFrames", str(WARMUP_FRAMES),
        "-RecordFrames", str(RECORD_FRAMES),
        "-Output", str(output_path),
    ]
    if update_size > 0:
        cmd += ["-UpdateSize", str(update_size)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  [FAIL] {output_name} -- exit code {result.returncode}")
            if result.stderr:
                print(f"    stderr: {result.stderr[-200:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [HANG] {output_name} -- timed out after 120s")
        return False


def count_tests():
    count = 0
    for grid in GRID_SIZES:
        valid_chunks = [c for c in CHUNK_SIZES if c <= grid // 2]
        for chunk in valid_chunks:
            for density in DENSITIES:
                for scheme in SCHEMES:
                    for seed in SEEDS:
                        count += 1
    for update_size in [1, 4, 16]:
        for scheme in SCHEMES:
            for seed in SEEDS:
                count += 1
    return count


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    total = count_tests()
    print(f"Total tests: {total}")
    print(f"Estimated runtime: ~{total * 22 / 3600:.1f} hours")
    print(f"Output directory: {OUTPUT_DIR.absolute()}\n")

    succeeded, failed, skipped = 0, 0, 0
    start_time = time.time()

    for grid in GRID_SIZES:
        valid_chunks = [c for c in CHUNK_SIZES if c <= grid // 2]
        for chunk in valid_chunks:
            for density in DENSITIES:
                for scheme in SCHEMES:
                    for seed in SEEDS:
                        name = f"g{grid}_c{chunk}_d{density}_s{scheme}_seed{seed}"
                        done = succeeded + failed + skipped
                        print(f"[{done+1}/{total}] {name} ...", end=" ", flush=True)
                        ok = run_test(grid, chunk, density, scheme, seed)
                        if ok:
                            succeeded += 1
                        else:
                            failed += 1
                        elapsed = time.time() - start_time
                        done = succeeded + failed + skipped
                        eta = elapsed / done * (total - done) if done > 0 else 0
                        print(f"[{'OK' if ok else 'FAIL'}] ETA: {eta/3600:.1f}h")

    for update_size in [1, 4, 16]:
        for scheme in SCHEMES:
            for seed in SEEDS:
                name = f"g128_c8_d50_s{scheme}_seed{seed}_u{update_size}"
                done = succeeded + failed + skipped
                print(f"[{done+1}/{total}] {name} ...", end=" ", flush=True)
                ok = run_test(128, 8, 50, scheme, seed, update_size)
                if ok:
                    succeeded += 1
                else:
                    failed += 1

    elapsed = time.time() - start_time
    print(f"\nDone. {succeeded} succeeded, {failed} failed, {skipped} skipped in {elapsed/3600:.1f}h")


if __name__ == "__main__":
    main()
