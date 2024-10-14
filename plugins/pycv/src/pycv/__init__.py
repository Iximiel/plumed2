import inspect
# checked and formatted with ruff 0.6.9


def getPythonCVInterface():
    import pycv

    path_of_this = inspect.getfile(pycv)
    path_of_lib = path_of_this[: path_of_this.rfind("/")]
    return path_of_lib + "/PythonCVInterface.so"


def main():
    print(getPythonCVInterface())
    return 0
