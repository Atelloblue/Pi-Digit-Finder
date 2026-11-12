#!/usr/bin/env python3
"""
Fast multi-core Chudnovsky pi (gmpy2).

Usage:
    python pi_fast.py 10000000          # one-shot, writes Pi.txt
    python pi_fast.py 10000000 8        # force 8 workers
    python pi_fast.py                   # stepping benchmark mode
"""
import gc
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
import multiprocessing as mp

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

try:
    import gmpy2
    from gmpy2 import mpz, isqrt
except ImportError:
    sys.stderr.write("\nERROR: gmpy2 is required.  pip install gmpy2\n\n")
    sys.exit(1)

# gmpy2 contexts are THREAD-LOCAL: the flag must be set inside every thread
# that does big-number math, otherwise those threads keep the GIL and run
# serially.  (Setting it only in the main thread does nothing for pool threads.)
HAS_GIL_RELEASE = hasattr(gmpy2.get_context(), "allow_release_gil")


def enable_gil_release():
    if HAS_GIL_RELEASE:
        gmpy2.get_context().allow_release_gil = True


enable_gil_release()

K = 640320 ** 3 // 24            # 10939058860032000
A, B = 13591409, 545140134
DIGITS_PER_TERM = 14.1816474627254776555
LEAF = 24                        # terms evaluated in a flat loop (tuned)
SPLIT_MIN_BITS = 1 << 21         # don't split multiplies smaller than this
LOG2_10 = 3.3219280948873626


# ----------------------------------------------------------------------------
# Binary splitting
# ----------------------------------------------------------------------------
def _leaf(a, b):
    """Sequential accumulate of terms [a, b) with small native ints."""
    P, Q, T = 1, 1, 0
    for k in range(a, b):
        if k == 0:
            p, q, t = 1, 1, A
        else:
            p = (6 * k - 5) * (2 * k - 1) * (6 * k - 1)
            q = k * k * k * K
            t = p * (A + B * k)
            if k & 1:
                t = -t
        T = T * q + P * t
        P *= p
        Q *= q
    return mpz(P), mpz(Q), mpz(T)


def bs(a, b, need_p):
    """(P, Q, T) over [a, b).  P is skipped (None) on the right spine."""
    if b - a <= LEAF:
        P, Q, T = _leaf(a, b)
        return (P if need_p else None), Q, T
    m = (a + b) >> 1
    p1, q1, t1 = bs(a, m, True)
    p2, q2, t2 = bs(m, b, need_p)
    return (p1 * p2 if need_p else None), q1 * q2, t1 * q2 + p1 * t2


def bs_task(args):
    return bs(*args)


def sqrt_task(calc_digits):
    return isqrt(10005 * mpz(10) ** (2 * calc_digits))


def conv_task(args):
    x, width, pad = args
    s = x.digits(10)
    return s.zfill(width) if pad else s


def worker_init():
    gc.disable()


# ----------------------------------------------------------------------------
# Parallel big multiplication (main thread orchestrates, threads multiply)
# ----------------------------------------------------------------------------
class Inline:
    """Executor stand-in for the single-worker case."""
    def submit(self, fn, *args):
        f = Future()
        f.set_result(fn(*args))
        return f


def _mul(x, y):
    return x * y


class PMul:
    """x*y, optionally split into independent partial products."""
    __slots__ = ("futs", "sh", "neg")

    def __init__(self, ex, x, y, pieces):
        self.neg = (x < 0) != (y < 0)
        x, y = abs(x), abs(y)
        if x.bit_length() > y.bit_length():
            x, y = y, x                       # split the larger operand
        bl = y.bit_length()
        pieces = max(1, min(pieces, bl // SPLIT_MIN_BITS))
        if pieces == 1:
            self.sh = 0
            self.futs = [ex.submit(_mul, x, y)]
        else:
            sh = -(-bl // pieces)
            self.sh = sh
            parts = []
            for i in range(pieces):
                part = y >> (i * sh)
                if i < pieces - 1:
                    part = gmpy2.f_mod_2exp(part, sh)
                parts.append(part)
            self.futs = [ex.submit(_mul, x, p) for p in parts]

    def result(self):
        r = self.futs[0].result()
        for i in range(1, len(self.futs)):
            r += self.futs[i].result() << (i * self.sh)
        return -r if self.neg else r


def merge_level(ex, nodes, workers):
    """Merge adjacent pairs; all 3-4 products per pair run concurrently."""
    npairs = len(nodes) // 2
    jobs = sum(3 if nodes[2 * i + 1][0] is None else 4 for i in range(npairs))
    pieces = -(-workers // jobs)

    pend = []
    for i in range(npairs):
        p1, q1, t1 = nodes[2 * i]
        p2, q2, t2 = nodes[2 * i + 1]
        pend.append((
            PMul(ex, p1, p2, pieces) if p2 is not None else None,
            PMul(ex, q1, q2, pieces),
            PMul(ex, t1, q2, pieces),
            PMul(ex, p1, t2, pieces),
        ))
    out = []
    for pp, pq, pt1, pt2 in pend:
        out.append((pp.result() if pp else None,
                    pq.result(),
                    pt1.result() + pt2.result()))
    if len(nodes) & 1:
        out.append(nodes[-1])
    return out


# ----------------------------------------------------------------------------
# Parallel radix-10 conversion: split by powers of 10, convert pieces in procs
# ----------------------------------------------------------------------------
def to_decimal_parts(x, nd, pool, ex, workers):
    """Returns list of strings that concatenate to the nd-digit decimal of x."""
    if workers <= 1 or nd < 3_000_000:
        return [x.digits(10)]

    pieces = 1
    while pieces < workers:
        pieces *= 2
    w = -(-nd // pieces)

    # powers of ten needed: w * 2^j
    ks = []
    k = w
    while k < w * pieces:
        ks.append(k)
        k *= 2
    pows = dict(zip(ks, ex.map(lambda e: mpz(10) ** e, ks)))

    items = [(x, pieces, nd)]                 # (value, piece_count, digits)
    while any(pc > 1 for _, pc, _ in items):
        futs = []
        for v, pc, d in items:
            if pc == 1:
                futs.append(None)
                continue
            lo_pc = pc // 2
            futs.append(ex.submit(divmod, v, pows[lo_pc * w]))
        new = []
        for (v, pc, d), f in zip(items, futs):
            if f is None:
                new.append((v, pc, d))
                continue
            lo_pc = pc // 2
            hi, lo = f.result()
            new.append((hi, pc - lo_pc, d - lo_pc * w))
            new.append((lo, lo_pc, lo_pc * w))
        items = new

    tasks = [(v, d, i > 0) for i, (v, _, d) in enumerate(items)]
    return pool.map(conv_task, tasks, chunksize=1)


# ----------------------------------------------------------------------------
# Pi
# ----------------------------------------------------------------------------
def compute_pi(digits, pool, ex, workers):
    """Returns list of decimal-string parts of floor(pi * 10^digits)."""
    calc_digits = digits + 10
    n_terms = int(calc_digits / DIGITS_PER_TERM) + 1
    gc.disable()

    if workers <= 1 or n_terms < 4000:
        _, Q, T = bs(0, n_terms, False)
        sqrt_term = sqrt_task(calc_digits)
    else:
        chunks = min(workers * 2, n_terms // 1000)
        step = n_terms // chunks
        ranges = [(i * step, n_terms if i == chunks - 1 else (i + 1) * step,
                   i != chunks - 1) for i in range(chunks)]
        a_sqrt = pool.apply_async(sqrt_task, (calc_digits,))
        a_bs = pool.map_async(bs_task, ranges, chunksize=1)
        nodes = a_bs.get()
        while len(nodes) > 1:
            nodes = merge_level(ex, nodes, workers)
        _, Q, T = nodes[0]
        sqrt_term = a_sqrt.get()

    # Q and T are ~2x longer than the digits we need.  Drop the excess low
    # bits (same shift for both => ratio preserved) so the final division is
    # n-digit sized instead of 2n-digit sized.
    keep = int(calc_digits * LOG2_10) + 192
    s = T.bit_length() - keep
    if s > 0:
        Q >>= s
        T >>= s

    pi_int = (sqrt_term * 426880 * Q) // T          # pi * 10^calc_digits
    pi_int //= mpz(10) ** 10                         # drop guard digits
    del Q, T, sqrt_term

    parts = to_decimal_parts(pi_int, digits + 1, pool, ex, workers)
    gc.enable()
    return parts


def stream_save_pi(filename, parts):
    with open(filename, "w", buffering=1 << 20) as f:
        first = parts[0]
        f.write(first[0])
        f.write(".")
        f.write(first[1:])
        for p in parts[1:]:
            f.write(p)


def read_last_precision(filename="last_precision.txt", default=100_000):
    try:
        with open(filename) as f:
            return int(f.read().strip())
    except Exception:
        return default


def write_last_precision(precision, filename="last_precision.txt"):
    with open(filename, "w") as f:
        f.write(str(precision))


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def make_pool(workers):
    if workers <= 1:
        return None, Inline()
    # Fork BEFORE creating threads (fork + threads is unsafe).
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()
    pool = ctx.Pool(workers, initializer=worker_init)
    ex = ThreadPoolExecutor(max_workers=workers, initializer=enable_gil_release)
    return pool, ex


def main():
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else available_cpus()
    workers = max(1, workers)
    print("Fast GMPY2 Pi engine")
    print(f"Workers: {workers} | GIL-release in threads: "
          f"{'yes' if HAS_GIL_RELEASE and workers > 1 else 'n/a'}")

    pool, ex = make_pool(workers)
    try:
        if len(sys.argv) > 1:
            target = int(sys.argv[1])
            print(f"\nTarget: {target:,} digits")
            t0 = time.perf_counter()
            parts = compute_pi(target, pool, ex, workers)
            el = time.perf_counter() - t0
            print(f"Calculated in {el:.3f}s ({target / el:,.0f} digits/sec)")
            stream_save_pi("Pi.txt", parts)
            print("Stored to Pi.txt")
            return

        precision = read_last_precision()
        step = 100_000
        print(f"Starting loop at {precision:,} digits (+{step:,} per step)")
        while True:
            try:
                print(f"\nComputing {precision:,} digits...")
                t0 = time.perf_counter()
                parts = compute_pi(precision, pool, ex, workers)
                el = time.perf_counter() - t0
                print(f"{precision:,} digits in {el:.3f}s "
                      f"({precision / el:,.0f} digits/sec)")
                stream_save_pi("Pi.txt", parts)
                write_last_precision(precision)
                precision += step
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
            except MemoryError:
                print("\nOut of memory.")
                break
    finally:
        if pool is not None:
            pool.terminate()
            ex.shutdown(wait=False)


if __name__ == "__main__":
    main()
