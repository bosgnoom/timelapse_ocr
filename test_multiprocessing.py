""" Testing multiprocessing """

import multiprocessing
import time
from itertools import repeat
import timeit


def doet_het_werk(nr, arguments):
    a, b = arguments
    antwoord = "Testing... {}-{}-{}".format(nr, a, b)
    print(antwoord)
    time.sleep(1)
    return antwoord


def main():
    piep_piep = ["Ja", "Nee", "Misschien", "Nog wat", "zo meer", "laatste"]

    with multiprocessing.Pool() as pool:
        results = pool.starmap(doet_het_werk, enumerate(zip(piep_piep, repeat("x"))))
    print(results)


if __name__ == "__main__":
    main()