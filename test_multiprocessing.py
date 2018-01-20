""" Testing multiprocessing """

import multiprocessing
import time
import os
import timeit


def doet_het_werk(a, b):
    antwoord = "Testing... {}-{}-{}".format(a, b, multiprocessing.current_process().name)
    print(antwoord)
    time.sleep(1)
    return antwoord


def main():
    piep_piep = ["Ja", "Nee", "Misschien", "Nog wat", "zo meer", "laatste"]
    supr = [[x, "iets"] for x in piep_piep]

    with multiprocessing.Pool() as pool:
        results = pool.starmap(doet_het_werk, supr)
    print(results)


if __name__ == "__main__":
    print(timeit.timeit(main, number=1))