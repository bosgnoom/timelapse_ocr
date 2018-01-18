""" Testing multiprocessing """

import multiprocessing
import time
import os


def doet_het_werk(a, b):
    antwoord = "Testing... {}-{}-{}".format(a, os.getpid(), multiprocessing.current_process())
    return antwoord


if __name__ == "__main__":
    piep_piep = ["Ja", "Nee", "Misschien"]
    supr = [ [x, "iets"] for x in piep_piep ]

    with multiprocessing.Pool() as pool:
          results = pool.starmap(doet_het_werk, supr)
    print(results)
