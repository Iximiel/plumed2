import plumedCommunications as PLMD
import numpy

log = open("pydist.log", "w")

print("Imported my pydist+.", file=log)


def init(action: PLMD.PythonCVInterface):
    return {"Value": {"period": ["0", "1.3"]}}


def mypytest(action: PLMD.PythonCVInterface):
    ret = [action.getStep()]

    return ret
