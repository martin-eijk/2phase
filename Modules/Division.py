
"""
In this module the general help functions are defined.
"""


# Help functions
def safe_div(top, bot):
    """
    This function is made to prevent division by zero

    The input is (top, bot) resulting in:
    top/bot
    """

    # Prevent to have NaN
    if bot == 0:
        tt = 0
    else:
        tt = top/bot

    return tt
