from decimal import Decimal, getcontext
import os

def calculate_pi(precision):
    # Set the precision for Decimal calculations
    getcontext().prec = precision + 2  # extra digits for accuracy
    C = 426880 * Decimal(10005).sqrt()
    M = 1
    L = 13591409
    X = 1
    K = 6
    S = L

    for n in range(1, precision):
        M = (K**3 - 16 * K) * M // n**3
        L += 545140134
        X *= -262537412640768000
        S += Decimal(M * L) / X
        K += 12

    pi = C / S
    return str(pi)[:-1]  # remove the last digit for accuracy

def read_last_precision():
    """Read the last precision from a file."""
    try:
        if os.path.exists("last_precision.txt"):
            with open("last_precision.txt", "r") as file:
                return int(file.read().strip())
    except FileNotFoundError:
        print("last_precision.txt not found. Starting from precision 1.")
    except ValueError:
        print("Invalid value in last_precision.txt. Starting from precision 1.")
    except Exception as e:
        print(f"Unexpected error while reading last precision: {e}")

    return 1  # Start from 1 if the file doesn't exist

def write_last_precision(precision):
    """Write the current precision to a file."""
    try:
        with open("last_precision.txt", "w") as file:
            file.write(str(precision))
    except IOError as e:
        print(f"IO error while writing last precision: {e}")
    except Exception as e:
        print(f"Unexpected error while writing last precision: {e}")

def main():
    precision = read_last_precision()  # Read the last precision
    print(f"Resuming from precision: {precision}")

    while True:
        try:
            pi_value = calculate_pi(precision)
            with open("Pi.txt", "w") as file:
                file.write(pi_value)
            print(f"Calculated π to {precision} digits: {pi_value}")
            write_last_precision(precision)  # Save the current precision
            precision += 1  # Increase precision for the next calculation
        except MemoryError:
            print("Memory error during calculation. Please reduce the precision.")
            break
        except Exception as e:
            print(f"Unexpected error during calculation: {e}")
            break  # Exit the loop on error

if __name__ == "__main__":
    main()
