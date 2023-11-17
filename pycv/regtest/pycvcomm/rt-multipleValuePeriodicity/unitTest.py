import plumedCommunications as PLMD
import numpy

log = open("pydist.log", "w")

print("Imported my pydist+.", file=log)


def init(action: PLMD.PythonCVInterface):
    return {"Periodic": {"period": ["0", "1.3"]},
    "PeriodicPI": {"period": ["0", "pi"]}}


def mypytest(action: PLMD.PythonCVInterface):
    ret = {"nonPeriodic": action.getStep(), "Periodic": action.getStep(),"PeriodicPI": action.getStep()}

    return ret
