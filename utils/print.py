from sys import stdout


def print_peforming_task(task: str):
    print_block("Performing %s ..." % (task))


def print_taks_done(task: str):
    print_block("%s Done!" % (task))


def print_block(content: str, title: str = "", num_marks: int = 20):
    upper = ("=" * num_marks) + title + ("=" * num_marks)
    bottom = "=" * len(upper)
    stdout.write("\n" + upper + "\n" + "| %s " % (content) + "\n" + bottom + "\n")
    stdout.flush()


def print_title(title: str, num_marks: int = 20):
    upper = ("=" * num_marks) + title + ("=" * num_marks)
    stdout.write(upper + "\n")
    stdout.flush()


def print_percentages(prefix: str, percentage: float, icon: str = "="):
    stdout.write(
        "%s [%-20s] %d%%" % (prefix, icon * int(20 * percentage), percentage * 100)
    )


def replace_print_flush(string: str):
    stdout.write("\r")
    stdout.write(string)
    stdout.flush()


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter

