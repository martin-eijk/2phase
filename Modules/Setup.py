# Packages
from itertools import groupby
import sys
import numpy as np
import math as mt

# Additional functions
from Modules.Division import safe_div

"""
Generation of input data to start the simulation; body shapes, grid properties
"""


#                                       Cases
def standing_wave(pos, tan_dir_pos, norm_dir_spacing, mean_height, amp, cos_sin, wave_length, phase, fraction):
    """
    pos: position of
    norm_dir: 1 for y-direction 0 to x-direction
    tan_dir_pos: center positions of the cells in tangential direction of the free surface
    norm_dir_spacing: size of the cells in normal direction of the free surface
    grid_length: length of the free surface elements
    mean_height: of the free surface
    amp: the initial amplitude of the wave
    cos_sin: cosine wave [1] or sine wave [0]
    wave_length: in meters
    """

    # The normal direction is one for y-direction and 0 for x-direction
    norm_dir = ((pos==1)+(pos==3))

    # Complete volume fraction field equal to one
    fraction[2:np.size(fraction, 0) - 2, 2:np.size(fraction, 1) - 2] = 1

    # Empty vector b with length of free surface, gets filled by height values of the free surface
    b = np.zeros([np.size(fraction, norm_dir)])
    b[2: np.size(fraction, norm_dir) - 2] = mean_height - (amp * cos_sin * np.cos(2 * mt.pi / wave_length * tan_dir_pos
                                                                                  + phase*mt.pi)
                                                           + amp * (1 - cos_sin) * np.sin(2 * mt.pi / wave_length
                                                                                          * tan_dir_pos + phase*mt.pi))

    # Depending on the input pos which side gets filled by water, the vector gets substracted by the spacing until end
    if pos == 2 or pos == 3:
        for i in range(2, np.size(fraction, norm_dir) - 2):
            for j in range(2, np.size(fraction, 1 - norm_dir) - 2):
                if b[i] - norm_dir_spacing[j] > 0:
                    fraction[j * norm_dir + i * (1 - norm_dir), i * norm_dir + j * (1 - norm_dir)] = 0
                    b[i] = b[i] - norm_dir_spacing[j]
                elif b[i] > 0:
                    fraction[j * norm_dir + i * (1 - norm_dir), i * norm_dir + j * (1 - norm_dir)] = 1 - b[i] / \
                                                                                                     norm_dir_spacing[j]
                    b[i] = b[i] - norm_dir_spacing[j]
    elif pos == 4 or pos == 1:
        for i in reversed(range(2, np.size(fraction, norm_dir) - 2)):
            for j in reversed(range(2, np.size(fraction, 1 - norm_dir) - 2)):
                if b[i] - norm_dir_spacing[j] > 0:
                    fraction[j * norm_dir + i * (1 - norm_dir), i * norm_dir + j * (1 - norm_dir)] = 0
                    b[i] = b[i] - norm_dir_spacing[j]
                elif b[i] > 0:
                    fraction[j * norm_dir + i * (1 - norm_dir), i * norm_dir + j * (1 - norm_dir)] = 1 - b[i] / \
                                                                                                     norm_dir_spacing[j]
                    b[i] = b[i] - norm_dir_spacing[j]
    return fraction
def circle(domain, domainC, grid):
    """
    With this function a cylindrical shape body can be produced.
    The input grid contains the amount of elements in horizontal and vertical direction.
    The domain is the size of the computational domain.
    The domainC has the size of the wedge, a vector with with three inputs (x,y,r).
    """

    # A center column is needed so (odd number)
    N = int(grid[0, 0])
    M = int(grid[0, 1])
    domainC = np.asarray(domainC)

    dx = np.zeros([N + 4])
    dy = np.zeros([M + 4])
    x = np.zeros([N + 1])
    y = np.zeros([M + 1])

    for i in range(0, N + 1):
        x[i] = domain[0, 0] * i / N

    for j in range(0, M + 1):
        y[j] = domain[0, 1] * j / M


    dx[2: N + 2] = x[1: N + 1] - x[0: N]
    dy[2: M + 2] = y[1: M + 1] - y[0: M]

    dx[0: 2] = dx[2]
    dy[0: 2] = dy[2]
    dx[N + 2: N + 4] = dx[N + 1]
    dy[M + 2: M + 4] = dy[M + 1]

    placeC1x, placeC3x, placeC2x, placeC4x = [], [], [], []
    placeC1y, placeC3y, placeC2y, placeC4y = [], [], [], []

    for i in range(0, N + 1):
        if x[i] >= domainC[0] and x[i] <= domainC[0] + domainC[2]:
            for j in range(0, M):
                if y[j] <= np.round(np.sqrt(abs(domainC[2] ** 2 - (x[i] - domainC[0]) ** 2)) + domainC[1], 8) and \
                        y[j + 1] >= np.round(np.sqrt(abs(domainC[2] ** 2 - (x[i] - domainC[0]) ** 2)) + domainC[1], 8):
                    placeC1x.append(
                        [x[i], np.round(np.sqrt(abs(domainC[2] ** 2 - (x[i] - domainC[0]) ** 2)) + domainC[1], 8),
                         i + 2, j + 2])
                if y[j] <= np.round(-np.sqrt(abs(domainC[2] ** 2 - (x[i] - domainC[0]) ** 2)) + domainC[1], 8) and \
                        y[j + 1] >= np.round(-np.sqrt(abs(domainC[2] ** 2 - (x[i] - domainC[0]) ** 2)) + domainC[1], 8):
                    placeC2x.append(
                        [x[i], np.round(-np.sqrt(abs(domainC[2] ** 2 - (x[i] - domainC[0]) ** 2)) + domainC[1], 8),
                         i + 2, j + 2])
        elif x[i] <= domainC[0] and x[i] >= domainC[0] - domainC[2]:
            for j in range(0, M):
                if y[j] <= np.round(-np.sqrt(abs(domainC[2] ** 2 - (x[i] - domainC[0]) ** 2)) + domainC[1], 8) and \
                        y[j + 1] >= np.round(-np.sqrt(abs(domainC[2] ** 2 - (x[i] - domainC[0]) ** 2)) + domainC[1], 8):
                    placeC3x.append(
                        [x[i], np.round(-np.sqrt(abs(domainC[2] ** 2 - (x[i] - domainC[0]) ** 2)) + domainC[1], 8),
                         i + 2, j + 2])
                if y[j] <= np.round(np.sqrt(abs(domainC[2] ** 2 - (x[i] - domainC[0]) ** 2)) + domainC[1], 8) and \
                        y[j + 1] >= np.round(np.sqrt(abs(domainC[2] ** 2 - (x[i] - domainC[0]) ** 2)) + domainC[1], 8):
                    placeC4x.append(
                        [x[i], np.round(np.sqrt(abs(domainC[2] ** 2 - (x[i] - domainC[0]) ** 2)) + domainC[1], 8),
                         i + 2, j + 2])

    for j in range(0, M + 1):
        if y[j] >= domainC[1] and y[j] <= domainC[1] + domainC[2]:
            for i in range(0, N):
                if x[i] <= np.round(np.sqrt(abs(domainC[2] ** 2 - (y[j] - domainC[1]) ** 2)) + domainC[0], 8) and \
                        x[i + 1] >= np.round(np.sqrt(abs(domainC[2] ** 2 - (y[j] - domainC[1]) ** 2)) + domainC[0], 8):
                    placeC1y.append(
                        [np.round(np.sqrt(abs(domainC[2] ** 2 - (y[j] - domainC[1]) ** 2)) + domainC[0], 8), y[j],
                         i + 2, j + 2])
                if x[i] <= np.round(-np.sqrt(abs(domainC[2] ** 2 - (y[j] - domainC[1]) ** 2)) + domainC[0], 8) and \
                        x[i + 1] >= np.round(-np.sqrt(abs(domainC[2] ** 2 - (y[j] - domainC[1]) ** 2)) + domainC[0], 8):
                    placeC4y.append(
                        [np.round(-np.sqrt(abs(domainC[2] ** 2 - (y[j] - domainC[1]) ** 2)) + domainC[0], 8), y[j],
                         i + 2, j + 2])
        elif y[j] <= domainC[1] and y[j] >= domainC[1] - domainC[2]:
            for i in range(0, N):
                if x[i] <= np.round(-np.sqrt(abs(domainC[2] ** 2 - (y[j] - domainC[1]) ** 2)) + domainC[0], 8) and \
                        x[i + 1] >= np.round(-np.sqrt(abs(domainC[2] ** 2 - (y[j] - domainC[1]) ** 2)) + domainC[0], 8):
                    placeC3y.append(
                        [np.round(-np.sqrt(abs(domainC[2] ** 2 - (y[j] - domainC[1]) ** 2)) + domainC[0], 8), y[j],
                         i + 2, j + 2])
                if x[i] <= np.round(np.sqrt(abs(domainC[2] ** 2 - (y[j] - domainC[1]) ** 2)) + domainC[0], 8) and \
                        x[i + 1] >= np.round(np.sqrt(abs(domainC[2] ** 2 - (y[j] - domainC[1]) ** 2)) + domainC[0], 8):
                    placeC2y.append(
                        [np.round(np.sqrt(abs(domainC[2] ** 2 - (y[j] - domainC[1]) ** 2)) + domainC[0], 8), y[j],
                         i + 2, j + 2])

    if len(placeC1x) != 0:
        placeC1x = np.asarray(placeC1x)
        placeC1x = placeC1x[placeC1x[:, 0].argsort()]
    if len(placeC2x) != 0:
        placeC2x = np.asarray(placeC2x)
        placeC2x = placeC2x[placeC2x[:, 0].argsort()]
    if len(placeC3x) != 0:
        placeC3x = np.asarray(placeC3x)
        placeC3x = placeC3x[placeC3x[:, 0].argsort()]
    if len(placeC4x) != 0:
        placeC4x = np.asarray(placeC4x)
        placeC4x = placeC4x[placeC4x[:, 0].argsort()]

    if len(placeC1y) != 0:
        placeC1y = np.asarray(placeC1y)
        placeC1y = placeC1y[placeC1y[:, 0].argsort()]
    if len(placeC2y) != 0:
        placeC2y = np.asarray(placeC2y)
        placeC2y = placeC2y[placeC2y[:, 0].argsort()]
    if len(placeC3y) != 0:
        placeC3y = np.asarray(placeC3y)
        placeC3y = placeC3y[placeC3y[:, 0].argsort()]
    if len(placeC4y) != 0:
        placeC4y = np.asarray(placeC4y)
        placeC4y = placeC4y[placeC4y[:, 0].argsort()]

    # It is assumed that not two lines are going through the same face and knowing at which side the wedge is
    Ax = np.zeros([M + 4, N + 5])
    Ay = np.zeros([M + 5, N + 4])

    Fb = np.zeros([M + 4, N + 4])

    for i in range(2, N + 2):
        for j in range(2, M + 2):
            if (((x[i - 1]+x[i-2])/2 - domainC[0]) ** 2 + (y[j - 2] - domainC[1]) ** 2) <= domainC[2] ** 2:
                Ay[j, i] = 1
    for i in range(2, N + 2):
        for j in range(2, M + 2):
            if ((x[i - 2] - domainC[0]) ** 2 + ((y[j - 1]+y[j-2])/2 - domainC[1]) ** 2) < domainC[2] ** 2:
                Ax[j, i] = 1
    for i in range(2, N + 2):
        for j in range(2, M + 2):
            if (((x[i - 1]+x[i-2])/2 - domainC[0]) ** 2 + ((y[j - 1]+y[j-2])/2 - domainC[1]) ** 2) <= domainC[2] ** 2:
                Fb[j, i] = 1

    for g in range(0, np.size(placeC3x, 0)):
        i, j = int(placeC3x[g, 2]), int(placeC3x[g, 3])
        kk = 1
        Ax[j, i] = abs(kk - abs(placeC3x[g, 1] - y[j - 2]) / abs(y[j - 1] - y[j - 2]))

    for g in range(0, np.size(placeC1x, 0)):
        i, j = int(placeC1x[g, 2]), int(placeC1x[g, 3])
        kk = 0
        Ax[j, i] = abs(kk - abs(placeC1x[g, 1] - y[j - 2]) / abs(y[j - 1] - y[j - 2]))

    for g in range(0, np.size(placeC2x, 0)):
        i, j = int(placeC2x[g, 2]), int(placeC2x[g, 3])
        kk = 1
        Ax[j, i] = abs(kk - abs(placeC2x[g, 1] - y[j - 2]) / abs(y[j - 1] - y[j - 2]))

    for g in range(0, np.size(placeC4x, 0)):
        i, j = int(placeC4x[g, 2]), int(placeC4x[g, 3])
        kk = 0
        Ax[j, i] = abs(kk - abs(placeC4x[g, 1] - y[j - 2]) / abs(y[j - 1] - y[j - 2]))

    for g in range(0, np.size(placeC1y, 0)):
        i, j = int(placeC1y[g, 2]), int(placeC1y[g, 3])
        kk = 0
        Ay[j, i] = abs(kk - abs(placeC1y[g, 0] - x[i - 2]) / abs(x[i - 1] - x[i - 2]))

    for g in range(0, np.size(placeC2y, 0)):
        i, j = int(placeC2y[g, 2]), int(placeC2y[g, 3])
        kk = 0
        Ay[j, i] = abs(kk - abs(placeC2y[g, 0] - x[i - 2]) / abs(x[i - 1] - x[i - 2]))

    for g in range(0, np.size(placeC3y, 0)):
        i, j = int(placeC3y[g, 2]), int(placeC3y[g, 3])
        kk = 1
        Ay[j, i] = abs(kk - abs(placeC3y[g, 0] - x[i - 2]) / abs(x[i - 1] - x[i - 2]))

    for g in range(0, np.size(placeC4y, 0)):
        i, j = int(placeC4y[g, 2]), int(placeC4y[g, 3])
        kk = 1
        Ay[j, i] = abs(kk - abs(placeC4y[g, 0] - x[i - 2]) / abs(x[i - 1] - x[i - 2]))

    Ax = np.ones([M + 4, N + 5]) - Ax
    Ay = np.ones([M + 5, N + 4]) - Ay

    placeC = []

    for i in range(2, N + 2):
        for j in range(2, M + 2):
            if Ax[j, i] + Ax[j, i + 1] + Ay[j, i] + Ay[j + 1, i] < 4 - 1e-8 and Ax[j, i] + Ax[j, i + 1] + Ay[j, i] + Ay[
                j + 1, i] > 1e-8:
                placeC.append([j, i])

    placeC = np.asarray(placeC)
    placeC = placeC[placeC[:, 0].argsort()]

    for g in range(0, len(placeC)):
        i, j = int(placeC[g, 1]), int(placeC[g, 0])
        kk = 0
        if Ax[j, i] >= 0 and Ax[j, i] <= 1 and \
                Ax[j, i + 1] >= 0 and Ax[j, i + 1] <= 1 and \
                (Ay[j, i] < 1e-8 or Ay[j, i] > 1 - 1e-8) and \
                (Ay[j + 1, i] < 1e-8 or Ay[j + 1, i] > 1 - 1e-8):
            if Ay[j, i] < Ay[j + 1, i]:
                kk = 0
            else:
                kk = 1

            Fb[j, i] = abs(1 - min(Ax[j, i + 1], Ax[j, i]) - 0.5 * abs(Ax[j, i + 1] - Ax[j, i]))

        elif Ay[j, i] >= 0 and Ay[j, i] <= 1 and \
                Ay[j + 1, i] >= 0 and Ay[j + 1, i] <= 1 and \
                (Ax[j, i] < 1e-8 or Ax[j, i] > 1 - 1e-8) and \
                (Ax[j, i + 1] < 1e-8 or Ax[j, i + 1] > 1 - 1e-8):
            if Ax[j, i] < Ax[j, i + 1]:
                kk = 0
            else:
                kk = 1

            Fb[j, i] = abs(1 - min(Ay[j + 1, i], Ay[j, i]) - 0.5 * abs(Ay[j + 1, i] - Ay[j, i]))

        elif Ax[j, i] >= 0 and Ax[j, i] <= 1 and \
                Ay[j, i] >= 0 and Ay[j, i] <= 1 and \
                (Ax[j, i + 1] < 1e-8 or Ax[j, i + 1] > 1 - 1e-8) and \
                (Ay[j + 1, i] < 1e-8 or Ay[j + 1, i] > 1 - 1e-8):
            if Ay[j, i] >= Ay[j + 1, i]:
                kk = 0
            else:
                kk = 1

            Fb[j, i] = abs((1 - kk) - 0.5 * abs(kk - Ay[j, i]) * abs(kk - Ax[j, i]))

        elif Ax[j, i + 1] >= 0 and Ax[j, i + 1] <= 1 and \
                Ay[j + 1, i] >= 0 and Ay[j + 1, i] <= 1 and \
                (Ax[j, i] < 1e-8 or Ax[j, i] > 1 - 1e-8) and \
                (Ay[j, i] < 1e-8 or Ay[j, i] > 1 - 1e-8):
            if Ay[j, i] < Ay[j + 1, i]:
                kk = 0
            else:
                kk = 1

            Fb[j, i] = abs((1 - kk) - 0.5 * abs(kk - Ay[j + 1, i]) * abs(kk - Ax[j, i + 1]))

        elif Ax[j, i + 1] >= 0 and Ax[j, i + 1] <= 1 and \
                Ay[j, i] >= 0 and Ay[j, i] <= 1 and \
                (Ax[j, i] < 1e-8 or Ax[j, i] > 1 - 1e-8) and \
                (Ay[j + 1, i] < 1e-8 or Ay[j + 1, i] > 1 - 1e-8):
            if Ay[j, i] >= Ay[j + 1, i]:
                kk = 0
            else:
                kk = 1

            Fb[j, i] = abs((1 - kk) - 0.5 * abs(kk - Ay[j, i]) * abs(kk - Ax[j, i + 1]))
        elif Ax[j, i] >= 0 and Ax[j, i] <= 1 and \
                Ay[j + 1, i] >= 0 and Ay[j + 1, i] <= 1 and \
                (Ax[j, i + 1] < 1e-8 or Ax[j, i + 1] > 1 - 1e-8) and \
                (Ay[j, i] < 1e-8 or Ay[j, i] > 1 - 1e-8):
            if Ay[j, i] < Ay[j + 1, i]:
                kk = 0
            else:
                kk = 1

            Fb[j, i] = abs((1 - kk) - 0.5 * abs(kk - Ay[j + 1, i]) * abs(kk - Ax[j, i]))

    Fb = np.flipud(Fb)
    Ax = np.flipud(Ax)
    Ay = np.flipud(Ay)
    return Fb, Ax, Ay, domainC
def block(hor_min, hor_max, vert_min, vert_max, domain, fraction):
    """
    This function is used to produce water columns.
    With the input hor_min to hor_max determines the horizontal length in the domain; hor_min<hor_max<domain[0]
    Vice versa for the vertical size.
    """

    # Domain fraction for the water column
    vert_max, vert_min, hor_max, hor_min = vert_max / domain[0, 1], vert_min / domain[0, 1], hor_max / domain[0, 0], hor_min / \
                                           domain[0, 0]

    # Amount of grid cells in horizontal N and vertical M
    N = np.size(fraction, 1) - 4
    M = np.size(fraction, 0) - 4

    # Internal cells get filled which are not at the boundary of the column
    fraction[np.size(fraction, 0) - mt.floor(vert_max * M) - 2:np.size(fraction, 0) - mt.ceil(vert_min * M) - 2,
    2 + mt.ceil(hor_min * N):2 + mt.floor(hor_max * N)] = 1

    # Sides of the column
    fraction[np.size(fraction, 0) - mt.floor(vert_max * M) - 2:np.size(fraction, 0) - mt.ceil(vert_min * M) - 2,
    mt.floor(hor_max * N) + 2] = hor_max * N - mt.floor(hor_max * N)
    fraction[np.size(fraction, 0) - mt.floor(vert_max * M) - 3, 2 + mt.ceil(hor_min * N):mt.floor(hor_max * N) + 2] = \
        vert_max * M - mt.floor(vert_max * M)
    fraction[np.size(fraction, 0) - mt.floor(vert_max * M) - 2:np.size(fraction, 0) - mt.ceil(vert_min * M) - 2,
    mt.ceil(hor_min * N) + 1] = mt.ceil(hor_min * N) - hor_min * N
    fraction[np.size(fraction, 0) - mt.ceil(vert_min * M) - 2, 2 + mt.ceil(hor_min * N):mt.floor(hor_max * N) + 2] = \
        mt.ceil(vert_min * M) - vert_min * M

    # Corners of the column
    fraction[np.size(fraction, 0) - mt.floor(vert_max * M) - 3, mt.floor(hor_max * N) + 2] = \
        (vert_max * M - mt.floor(vert_max * M)) * (hor_max * N - mt.floor(hor_max * N))
    fraction[np.size(fraction, 0) - mt.ceil(vert_min * M) - 2, mt.ceil(hor_min * N) + 1] = \
        (mt.ceil(vert_min * M) - vert_min * M) * (mt.ceil(hor_min * N) - hor_min * N)
    fraction[np.size(fraction, 0) - mt.ceil(vert_min * M) - 2, mt.floor(hor_max * N) + 2] = \
        (mt.ceil(vert_min * M) - vert_min * M) * (hor_max * N - mt.floor(hor_max * N))
    fraction[np.size(fraction, 0) - mt.floor(vert_max * M) - 3, mt.ceil(hor_min * N) + 1] = \
        (vert_max * M - mt.floor(vert_max * M)) * (mt.ceil(hor_min * N) - hor_min * N)

    return fraction

#                                       Setup
# Subfunctions
def read_input():
    """
     In this function, all the input is readed from the 'input.tex' file.
     Furthermore, all empty lines and lines starting with '#' are removed from the input file.
     The data is changed into floats, except for the name.
    """

    # Empty input
    Input = []
    Data = []
    name = []

    # Read file
    with open(sys.path[0]+'/input.tex') as f_data:
        for k, g in groupby(f_data, lambda x: x.startswith(('#','\''))):
            if not k:
                Input.append(np.array([[float(x) for x in d.split()] for d in g if len(d.strip())]))

    with open(sys.path[0]+'/input.tex') as f_data:
        for k, g in groupby(f_data, lambda x: not x.startswith(('\''))):
            if not k:
                name.append(np.array([[str(x) for x in d.split()] for d in g if len(d.strip())]))

    # Remove empty lines
    for x in range(0, len(Input)):
        if np.size(Input[x]) > 0:
            Data.append(Input[x])

    return Data, name
def sort_input(Data, name):
    """
     In this function the readfile data is named and sorted
    """

    # Defining vectors
    name = np.asarray(name[0])
    name = name[0, 1]

    # Parameters
    Parameters1 = Data[0]
    Parameters2 = Data[2]
    grav_con = Data[1]
    atm_pres_side = Data[3]

    # Domain
    Domain = Data[4]
    Grid = Data[5]

    # Standing wave
    Case = Data[6]
    Wave = Data[7]
    cossin = Data[8]
    dirwave = Data[9]
    Case = int(Case[0, 0])
    cossin = int(cossin[0, 0])
    dirwave = int(dirwave[0, 0])

    # Free flow
    Freeflow11 = Data[10]
    Freeflow21 = Data[11]

    # Time values
    Time = Data[12]
    CFL = Data[13]
    CFLcrit = Data[14]

    # Type of simulation
    Type = Data[15:23]

    # Postprocessing
    Heights = Data[23]
    Pressures = Data[24]
    Mass_data = Data[25]
    Save_data = Data[26]
    Save_fig = Data[27]
    Save_VTK = Data[28]

    return Parameters1, Parameters2, grav_con, Domain, Grid, Freeflow11, Freeflow21, Time, CFL, \
           CFLcrit, Type, Heights, Pressures, Mass_data, Save_data, Save_fig, Save_VTK, name, atm_pres_side, Case, \
           Wave, cossin, dirwave
def grid_input(domain, grid, atm_pres):
    """
    In this function the arrays for the grid are generated as well as initial conditions.
    """

    # Amount of elements in direction
    N = int(grid[0, 0])
    M = int(grid[0, 1])

    # Empty arrays
    dx = np.zeros([N + 4])
    dy = np.zeros([M + 4])
    x_node = np.zeros([N + 1])
    y_node = np.zeros([M + 1])
    x = np.zeros([N + 1])
    y = np.zeros([M + 1])
    u = np.zeros([M + 4, N + 5])
    v = np.zeros([M + 5, N + 4])
    boundary = np.zeros([M + 4, N + 4])
    pres = np.zeros([M + 4, N + 4])
    pres[1:M + 3, 1:N + 3] = atm_pres

    # Cell centers
    for i in range(0, N + 1):
        x_node[i] = domain[0, 0] * i / N

    for j in range(0, M + 1):
        y_node[j] = domain[0, 1] * j / M

    # Spacing distances
    dx[2: N + 2] = x_node[1: N + 1] - x_node[0: N]
    dy[2: M + 2] = y_node[1: M + 1] - y_node[0: M]

    dx[0: 2] = dx[2]
    dy[0: 2] = dy[2]
    dx[N + 2: N + 4] = dx[N + 1]
    dy[M + 2: M + 4] = dy[M + 1]

    # Cell faces
    for i in range(0, np.size(x_node) - 1):
        x[i] = (x_node[i] + x_node[i + 1]) / 2

    for j in range(0, np.size(y_node) - 1):
        y[j] = (y_node[j] + y_node[j + 1]) / 2

    x = np.delete(x, np.size(x) - 1)
    y = np.delete(y, np.size(y) - 1)

    # 3D array cell centers
    z_node = [0, 1]
    x_node2 = np.zeros([N + 1, 2, M + 1])
    y_node2 = np.zeros([N + 1, 2, M + 1])
    z_node2 = np.zeros([N + 1, 2, M + 1])
    for k in range(2):
        for j in range(M + 1):
            for i in range(N + 1):
                x_node2[i, k, j] = x_node[i]
                y_node2[i, k, j] = y_node[j]
                z_node2[i, k, j] = z_node[k]

    return x, y, x_node2, y_node2, z_node2, dx, dy, u, v, boundary, pres

# Mainfunction
def input():
    """
     In this HEAD function, all the initial conditions, input values are named for the simulations.
    """

    # Load input
    Data, name = read_input()
    Parameters1, Parameters2, grav_con, domain, grid, Freeflow11, Freeflow21, \
    Time, CFL, CFLcrit, Type, Heights, Pressures, Mass_data, Save_data, Save_fig, Save_VTK, \
    name, atm_pres_side, Case, Wave, cos_sin, dirwave = sort_input(Data, name)

    # Define variables
    rho_con_a, rho_con_l, mu_con_a, mu_con_l = Parameters1[0, 0], Parameters1[0, 1], Parameters1[0, 2], \
                                               Parameters1[0, 3]
    sigma, atm_pres, gamma_a = Parameters2[0, 0], Parameters2[0, 1], Parameters2[0, 2]

    x, y, x_node, y_node, z_node, dx, dy, u, v, boundary, pres = grid_input(domain, grid, atm_pres)

    # Empty head arrays
    F1 = np.zeros([int(grid[0, 1])+4, int(grid[0, 0])+4])
    F2 = np.zeros([int(grid[0, 1]) + 4, int(grid[0, 0]) + 4])

    if Case == 0:
        # Flow field 1
        if Freeflow11[0, 1] == 3:
            F1 = block(Freeflow11[0, 2], Freeflow11[0, 3], Freeflow11[0, 4], Freeflow11[0, 5], domain, F1)
        elif Freeflow11[0, 1] == 2:
            F1, NT, NT, NT = circle(domain, [Freeflow11[0, 2], Freeflow11[0, 3], Freeflow11[0, 4]], grid)
        elif Freeflow11[0, 1] != 0:
            print('Wrong input for free flow 1')

        # Flow field 2
        if Freeflow21[0, 1] == 3:
            F2 = block(Freeflow21[0, 2], Freeflow21[0, 3], Freeflow21[0, 4], Freeflow21[0, 5], domain, F2)
        elif Freeflow21[0, 1] == 2:
            F2, NT, NT, NT = circle(domain, [Freeflow21[0, 2], Freeflow21[0, 3], Freeflow21[0, 4]], grid)
        elif Freeflow21[0, 1] != 0:
            print('Wrong input for free flow 2')

        # Air or Water
        if Freeflow11[0, 0] == 1:
            F1 = (1-F1)
        if Freeflow21[0, 0] == 1:
            F2 = (1-F2)

        # Boundaries
        F = np.clip(F1-F2, a_min = 0.0, a_max = 1.0)

    elif Case == 1:
        if dirwave == 1 or dirwave == 3:
            tan_dir_pos = x
            norm_dir_spacing = dy
        elif dirwave == 2 or dirwave == 4:
            tan_dir_pos = y
            norm_dir_spacing = dx
        else:
            print('Wrong input dirwave')
            exit()
        F = standing_wave(dirwave, tan_dir_pos, norm_dir_spacing, Wave[0, 1], Wave[0, 0],
                          cos_sin, Wave[0, 2], Wave[0, 3], F1)
    else:
        print('Wrong integer for Case')
        exit()

    # Time
    t, dt, dt_min, dt_max, iter = Time[0, 0], Time[0, 1], Time[0, 2], Time[0, 3], Time[0, 4]

    # Type
    UPW, AB, HF, GC, HN, COM, MCHS, SLIP = Type[0], Type[1], Type[2], Type[3], Type[4], \
                                                                   Type[5], Type[6], Type[7]
    UPW, AB, HF, GC, HN, COM, MCHS, SLIP= int(UPW[0, 0]), int(AB[0, 0]), int(HF[0, 0]), \
                                                                   int(GC[0, 0]), int(HN[0, 0]), \
                                                                   int(COM[0, 0]), int(MCHS[0, 0]), \
                                                                   int(SLIP[0, 0])

    return grid, boundary, x, y, x_node, y_node, z_node, dx, dy, u, v, pres, F, domain, t, dt, dt_min, dt_max, iter, CFL, CFLcrit, mu_con_l, \
           mu_con_a, rho_con_l, rho_con_a, gamma_a, Heights, Pressures, Mass_data, Save_data, Save_fig,\
           Save_VTK, sigma, grav_con, atm_pres, UPW, AB, HF, GC, HN, COM, MCHS, SLIP, name, atm_pres_side
