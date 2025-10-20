import os
import time
from mpmath import mp

# Try to enable gmpy2 backend for mpmath if available
try:
    import gmpy2
    print("✅ Using gmpy2 backend for faster calculations.")
except ImportError:
    print("⚙️ gmpy2 not found — using default mpmath backend.")

def calculate_pi(precision):
    mp.dps = precision
    return mp.nstr(mp.pi, n=precision)

def read_last_precision():
    """Read last saved precision from file."""
    if os.path.exists("last_precision.txt"):
        try:
            with open("last_precision.txt", "r") as f:
                return int(f.read().strip())
        except Exception:
            pass
    return 1000  # start at 1k digits if none saved

def write_last_precision(precision):
    """Save current precision to file."""
    with open("last_precision.txt", "w") as f:
        f.write(str(precision))

def main():
    precision = read_last_precision()
    print(f"Resuming from precision: {precision}")

    while True:
        start = time.time()
        try:
            pi_value = calculate_pi(precision)
            elapsed = time.time() - start
            print(f"Calculated π to {precision} digits in {elapsed:.2f}s")

            # Save every 5k digits to reduce disk I/O
            if precision % 5000 == 0:
                with open("Pi.txt", "w") as f:
                    f.write(pi_value)
                write_last_precision(precision)
                print(f"💾 Saved Pi.txt and last precision at {precision} digits.")

            # Increase precision in bigger jumps for performance
            precision += 1000
        except MemoryError:
            print("❌ Out of memory — try lowering precision or step size.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

if __name__ == "__main__":
    main()
