from IPython import get_ipython


class BasePlotter:
    def script_type(self) -> str:
        """Returns script type to determine if interactive plots are available."""
        if get_ipython() is not None:
            ipy_str = str(type(get_ipython()))
            if "zmqshell" in ipy_str:
                return "jupyter"
            if "terminal" in ipy_str:
                return "ipython"
        else:
            return "terminal"
