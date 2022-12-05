"""
The :mod:`py_fatigue.styling` module collects all the functions and
classes related with better displaying i/o operations.
"""

from typing import Optional, Type, Union

__all__ = ["TermColors", "py_fatigue_formatwarning"]


class TermColors:  # pragma: no cover
    """Color set for terminal outputs"""

    # pylint: disable=too-few-public-methods

    CEND = "\33[0m"
    CBOLD = "\33[1m"
    CITALIC = "\33[3m"
    CURL = "\33[4m"
    CBLINK = "\33[5m"
    CBLINK2 = "\33[6m"
    CSELECTED = "\33[7m"

    CBLACK = "\33[30m"
    CRED = "\33[31m"
    CGREEN = "\33[32m"
    CYELLOW = "\33[33m"
    CBLUE = "\33[34m"
    CVIOLET = "\33[35m"
    CBEIGE = "\33[36m"
    CWHITE = "\33[37m"

    CBLACKBG = "\33[40m"
    CREDBG = "\33[41m"
    CGREENBG = "\33[42m"
    CYELLOWBG = "\33[43m"
    CBLUEBG = "\33[44m"
    CVIOLETBG = "\33[45m"
    CBEIGEBG = "\33[46m"
    CWHITEBG = "\33[47m"

    CGREY = "\33[90m"
    CRED2 = "\33[91m"
    CGREEN2 = "\33[92m"
    CYELLOW2 = "\33[93m"
    CBLUE2 = "\33[94m"
    CVIOLET2 = "\33[95m"
    CBEIGE2 = "\33[96m"
    CWHITE2 = "\33[97m"

    CGREYBG = "\33[100m"
    CREDBG2 = "\33[101m"
    CGREENBG2 = "\33[102m"
    CYELLOWBG2 = "\33[103m"
    CBLUEBG2 = "\33[104m"
    CVIOLETBG2 = "\33[105m"
    CBEIGEBG2 = "\33[106m"
    CWHITEBG2 = "\33[107m"


def py_fatigue_formatwarning(
    message: Union[Warning, str],
    category: Type[Warning],
    filename: str,
    lineno: int,
    line: Optional[str] = None,
) -> str:  # pragma: no cover
    """Monkeypatch the warning messages.

    Parameters
    ----------
    message : Union[Warning, str]
        The warning message.
    category : Type[Warning]
        The warning category.
    filename : str
        The file name.
    lineno : int
        The warning line number.
    line : Optional[str], optional
        The warning line, by default None

    Returns
    -------
    str
        Monkeypatched warning message
    """
    # pylint: disable=unused-argument
    if category:
        w_msg = "".join(
            (
                f"{TermColors.CRED2}{TermColors.CBOLD}",
                f"{str(category.__name__,)}:\n{TermColors.CEND}",
                f"{TermColors.CYELLOW2}{message}{TermColors.CEND}\n",
            )
        )
        return w_msg
    return f"{TermColors.CYELLOW2}{message}{TermColors.CEND}\n"
