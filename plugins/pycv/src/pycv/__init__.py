# checked and formatted with ruff 0.6.9


def getPythonCVInterface():
    import inspect
    import pycv

    path_of_this = inspect.getfile(pycv)
    path_of_lib = path_of_this[: path_of_this.rfind("/")]
    return path_of_lib + "/PythonCVInterface.so"


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="pyCV",
        description="""shows the path for the pycv shared object.

        Just run this with no arguments to see the path.
        """,
    )
    _ = parser.parse_args()
    print(getPythonCVInterface())
    return 0
