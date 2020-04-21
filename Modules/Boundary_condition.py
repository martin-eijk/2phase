# Packages needed
import numpy as np

"""
In this module the boundary conditions for fractions and velocity in ghost cells are defined.
"""

# Functions
def slip_condition(vel, dir, slip, Nv, Mv):
    """
    With this function, the slip conditions in the ghost cells can be applied.

    :param vel: The velocity field
    :param dir: The corresponding direction of application (u=0, v=1)
    :param slip: The value of no-slip (-1) or free slip (-1)
    :return:
    """

    # Applying of no-slip or free-slip conditions
    # dir: x=0, y=0
    # slip: free=1, no=-1

    vel[1 + dir:2 * (1 - dir) + dir * Mv,
    2 - dir:(1 - dir) * Nv + 2 * dir] = slip * vel[2:3 * (1 - dir) + dir * Mv,
                                                                  2:(1 - dir) * Nv + 3 * dir]

    vel[2 * dir + (1 - dir) * Mv:Mv + 1 - dir,
    2 * (1 - dir) + dir * Nv:Nv + dir] = slip * vel[2 * dir + (1 - dir) * (Mv - 1):Mv,
                                                2 * (1 - dir) + dir * (Nv - 1):Nv]

    vel[0 + 2 * dir:1 * (1 - dir) + dir * Mv,
    2 - 2 * dir:1 * dir + (1 - dir) * Nv] = slip * vel[2:3 * (1 - dir) + dir * Mv,
                                                                      2:(1 - dir) * Nv + 3 * dir]
    vel[(1 - dir) * (Mv + 1) + dir * 2:Mv + 2 - 2 * dir,
    2 * (1 - dir) + dir * (Nv + 1):Nv + 2 - 2 * (1 - dir)] = slip * vel[2 * dir + (1 - dir) * (
                Mv - 1):Mv, 2 * (1 - dir) + dir * (Nv - 1):Nv]

    return vel
def fraction_condition(fraction, dir, slip, N, M):
    """
    With this function, the fraction value can be mirrored in the ghost cells

    :param j: vertical indices
    :param i: horizontal indices
    :param fraction: fraction field to apply
    :param dir: which axes to apply
    :param slip: equal to 0=empty or 1=mirroring
    :param cellslope_b: The orientation of cells of body
    :param p_b: positions indices of body boundary
    :return:
    """
    # At boundary
    fraction[1 + dir:2 * (1 - dir) + dir * M, 2 - dir:(1 - dir) * N + 2 * dir] = \
        slip * fraction[2:3 * (1 - dir) + dir * M, 2:(1 - dir) * N + 3 * dir]
    fraction[2 * dir + (1 - dir) * M:M + 1 - dir, 2 * (1 - dir) + dir * N:N + dir] = \
        slip * fraction[2 * dir + (1 - dir) * (M - 1):M, 2 * (1 - dir) + dir * (N - 1):N]

    return fraction
