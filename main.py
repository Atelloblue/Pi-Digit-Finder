import os
import sys
import time
import gc
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

# Disable string conversion limit for large numbers (Python 3.11+)
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

# --- STRICT REQUIREMENT: gmpy2 ---
try:
    import gmpy2
    from gmpy2 import mpz, isqrt
except ImportError:
    sys.stderr.write(
        "\n❌ ERROR: gmpy2 is required for this script.\n"
        "Install it via: pip install gmpy2\n\n"
    )
    sys.exit(1)

# Enable experimental GIL release for mpz operations in gmpy2 contexts
ctx = gmpy2.get_context()
try:
    ctx.allow_release_gil = True
except (AttributeError, TypeError):
    pass

# Chudnovsky series constants
C = 640320
C3_OVER_24 = C**3 // 24
C3_OVER_24_SQ = C3_OVER_24**2  # Precomputed for 2-leaf jumps
DIGITS_PER_TERM = 14.1816474627254776555

def worker_init():
    """Initializes worker processes with optimal arithmetic & memory settings."""
    gc.disable()
    try:
        gmpy2.get_context().allow_release_gil = True
    except (AttributeError, TypeError):
        pass

def bs_range(a: int, b: int):
    """
    Computes (P, Q, T) over [a, b) using binary splitting.
    Unrolled to base cases for b - a <= 2 using native 64-bit integer math.
    """
    diff = b - a
    if diff <= 2:
        if diff == 1:
            if a == 0:
                return mpz(1), mpz(1), mpz(13591409)
            p = (6 * a - 5) * (2 * a - 1) * (6 * a - 1)
            q = a * a * a * C3_OVER_24
            t = p * (13591409 + 545140134 * a)
            if a & 1:
                t = -t
            return mpz(p), mpz(q), mpz(t)

        # diff == 2: Unroll two adjacent terms simultaneously
        if a == 0:
            P = 5
            Q = C3_OVER_24
            T = 13591409 * C3_OVER_24 - 2793657715
            return mpz(P), mpz(Q), mpz(T)

        a1 = a + 1
        p1 = (6 * a - 5) * (2 * a - 1) * (6 * a - 1)
        p2 = (6 * a1 - 5) * (2 * a1 - 1) * (6 * a1 - 1)

        t1 = p1 * (13591409 + 545140134 * a)
        if a & 1:
            t1 = -t1

        t2 = p2 * (13591409 + 545140134 * a1)
        if a1 & 1:
            t2 = -t2

        q2 = (a1 * a1 * a1) * C3_OVER_24
        P = p1 * p2
        Q = ((a * a1) ** 3) * C3_OVER_24_SQ
        T = t1 * q2 + p1 * t2
        return mpz(P), mpz(Q), mpz(T)

    m = (a + b) // 2
    p1, q1, t1 = bs_range(a, m)
    p2, q2, t2 = bs_range(m, b)

    return p1 * p2, q1 * q2, t1 * q2 + p1 * t2

def merge_pqt(left, right):
    """Tree node merge with immediate dereferencing."""
    p1, q1, t1 = left
    p2, q2, t2 = right
    return p1 * p2, q1 * q2, t1 * q2 + p1 * t2

def merge_pair(pair):
    return merge_pqt(pair[0], pair[1])

def merge_pqt_final(left, right):
    """
    Root level merge: Skips the giant P = P1 * P2 multiplication completely
    since P is not needed once all terms are accumulated.
    """
    p1, q1, t1 = left
    _, q2, t2 = right
    Q = q1 * q2
    T = t1 * q2 + p1 * t2
    return Q, T

def compute_sqrt_term(calc_digits: int):
    """Runs concurrently in the background while workers process series terms."""
    scale = mpz(10) ** (2 * calc_digits)
    return isqrt(10005 * scale)

def choose_chunks(num_terms: int, num_workers: int) -> int:
    """Selects an optimal power-of-2 chunk count for parallel reduction."""
    if num_workers <= 1 or num_terms < 500:
        return 1
    chunks = 1
    while chunks < num_workers * 4 and (num_terms // (chunks * 2)) >= 250:
        chunks *= 2
    return max(1, chunks)

def compute_pi(digits: int, pool: Pool, num_workers: int) -> str:
    """Executes multi-core Chudnovsky computation and returns the raw digit string."""
    calc_digits = digits + 10  # 10 guard digits eliminate integer truncation error
    num_terms = int(calc_digits / DIGITS_PER_TERM) + 1
    chunks = choose_chunks(num_terms, num_workers)

    gc.disable()

    if chunks <= 1:
        # For small calculations, avoid process overhead entirely
        _, Q, T = bs_range(0, num_terms)
        sqrt_term = compute_sqrt_term(calc_digits)
    else:
        step = num_terms // chunks
        ranges = [
            (i * step, num_terms if i == chunks - 1 else (i + 1) * step)
            for i in range(chunks)
        ]

        # 1. Asynchronously compute square root on one worker
        async_sqrt = pool.apply_async(compute_sqrt_term, (calc_digits,))

        # 2. Dispatch series terms chunks across workers
        async_bs = pool.starmap_async(bs_range, ranges)

        # 3. Collect results (square root finishes during series computation)
        sqrt_term = async_sqrt.get()
        chunk_results = async_bs.get()

        # 4. Multi-threaded in-memory reduction (Zero IPC overhead)
        with ThreadPoolExecutor(max_workers=min(len(chunk_results) // 2, num_workers)) as tpool:
            while len(chunk_results) > 2:
                pairs = [(chunk_results[i], chunk_results[i + 1]) for i in range(0, len(chunk_results), 2)]
                chunk_results = list(tpool.map(merge_pair, pairs))

        # 5. Final merge with root multiplication pruning
        Q, T = merge_pqt_final(chunk_results[0], chunk_results[1])

    # Final division: Pi = (426880 * sqrt(10005) * Q) / T
    numerator = (sqrt_term * 426880) * Q
    pi_int = numerator // T

    # Fast subquadratic radix-10 string conversion via GMP's C-routine
    pi_digits = gmpy2.digits(pi_int)

    gc.enable()
    return pi_digits

def stream_save_pi(filename: str, pi_digits: str, digits: int):
    """Streams digits directly to disk in 1 MB blocks without copying giant strings in memory."""
    with open(filename, "w", buffering=1024 * 1024) as f:
        f.write(pi_digits[0])
        f.write(".")
        chunk_size = 1024 * 1024
        start = 1
        end = digits + 1
        while start < end:
            f.write(pi_digits[start : min(start + chunk_size, end)])
            start += chunk_size

def read_last_precision(filename="last_precision.txt", default=100_000):
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return int(f.read().strip())
        except Exception:
            pass
    return default

def write_last_precision(precision, filename="last_precision.txt"):
    with open(filename, "w") as f:
        f.write(str(precision))

def main():
    workers = max(1, cpu_count())
    print(f"🔥 Hyper-Optimized GMPY2 Pi Engine Started")
    print(f"⚙️ CPU Cores Allocated: {workers}")

    # One-shot mode: python script.py 5000000
    if len(sys.argv) > 1:
        target = int(sys.argv[1])
        print(f"\nTarget: {target:,} digits")
        with Pool(processes=workers, initializer=worker_init) as pool:
            t0 = time.perf_counter()
            pi_digits = compute_pi(target, pool, workers)
            elapsed = time.perf_counter() - t0

        rate = target / elapsed if elapsed > 0 else 0
        print(f"✨ Calculated in {elapsed:.3f}s ({rate:,.0f} digits/sec)")
        stream_save_pi("Pi.txt", pi_digits, target)
        print("💾 Stored to Pi.txt")
        return

    # Continuous benchmark / stepping mode
    precision = read_last_precision(default=100_000)
    step = 100_000
    print(f"Starting loop at {precision:,} digits (Stepping by +{step:,})")

    # Persistent worker pool avoids process spawn latency on each step
    with Pool(processes=workers, initializer=worker_init) as pool:
        while True:
            try:
                print(f"\nComputing {precision:,} digits...")
                t0 = time.perf_counter()

                pi_digits = compute_pi(precision, pool, workers)

                elapsed = time.perf_counter() - t0
                rate = precision / elapsed if elapsed > 0 else 0
                print(f"✅ {precision:,} digits in {elapsed:.3f}s ({rate:,.0f} digits/sec)")

                stream_save_pi("Pi.txt", pi_digits, precision)
                write_last_precision(precision)
                print(f"💾 Checkpointed to Pi.txt and last_precision.txt.")

                precision += step

            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
            except MemoryError:
                print("\n❌ Out of memory.")
                break

if __name__ == "__main__":
    main()
