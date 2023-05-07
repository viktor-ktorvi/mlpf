from enum import IntEnum


class BusTypeIds(IntEnum):
    """
    Enumeration for the ppc encoding of bus types.
    """

    PQ = 1
    PV = 2
    Slack = 3
