# Packages
import numpy as np

# Additional functions
from Modules.Variables import cell_labelling_variables
from Modules.Variables import reconstruction_variables

"""
This module represents the transport of the free surface.
"""


# Subfunctions
def Fad(j, i, dir, fraction, cellslope, h):
    """
    The application of Hirt&Nichols need a definition of F_ad.

    :param j: vertical indices
    :param i: horizontal indices
    :param dir: velocity field direction horizontal (0) or vertical (1)
    :param fraction: volume fraction field
    :param F_b: body fraction field
    :param cellslope: the orientation of S-cells
    :param h: the corresponding velocity field as in 'dir'
    :return:
    """
    Psi = 0
    flux_ir = fraction[j, i] + 0.5 * (fraction[j, i] - fraction[j - dir, i - 1 + dir]) * Psi

    Psi = 0
    flux_il = fraction[j - dir, i - 1 + dir] + 0.5 * (
            fraction[j - dir, i - 1 + dir] - fraction[j - 2 * dir, i - 2 + 2 * dir]) * Psi

    vel_dir = int((1 - np.sign(h)) / 2)

    if fraction[j - 2 * dir + vel_dir * dir, i - 2 + vel_dir * (1 - dir) + 2 * dir] == 0 or \
            fraction[j + vel_dir * dir, i + vel_dir * (1 - dir)] == 0 or \
            cellslope[j - dir + vel_dir * dir, i - 1 + dir + vel_dir * (1 - dir)] == 4 * (1 - dir) + dir or \
            cellslope[j - dir + vel_dir * dir, i - 1 + dir + vel_dir * (1 - dir)] == 2 * (1 - dir) + 3 * dir:
        Fad = flux_ir * (1 - vel_dir) + flux_il * vel_dir
        ID = (i * (1 - dir) + dir * j - 1) * (vel_dir) + (i * (1 - dir) + dir * j)*(1-vel_dir)
        IA = (i * (1 - dir) + dir * j) * (vel_dir) + (i * (1 - dir) + dir * j - 1)*(1-vel_dir)
    else:
        Fad = flux_il * (1 - vel_dir) + flux_ir * vel_dir
        ID = (i * (1 - dir) + dir * j - 1) * (1 - vel_dir) + (i * (1 - dir) + dir * j)*vel_dir
        IA = (i * (1 - dir) + dir * j) * (1 - vel_dir) + (i * (1 - dir) + dir * j - 1)*vel_dir

    return Fad, ID, IA
def order(m_vec_x, m_vec_y, spacing_hor, spacing_vert):
    """
    Determination of order of domination of normal vector on the orientation

    :param m_vec_x: normal vector components in x-direction
    :param m_vec_y: normal vector components in y-direction
    :param spacing_hor: horizontal spacing vector
    :param spacing_vert: vertical spacing vector
    :return:
    """
    if abs(m_vec_x * spacing_hor) < abs(m_vec_y * spacing_vert):
        m1 = abs(m_vec_x)
        dx1 = spacing_hor
        m2 = abs(m_vec_y)
        dx2 = spacing_vert
    else:
        m1 = abs(m_vec_y)
        dx1 = spacing_vert
        m2 = abs(m_vec_x)
        dx2 = spacing_hor

    return m1, m2, dx1, dx2
def cmpvolume(m_vec_x, m_vec_y, spacing_hor, spacing_vert, alpha):
    """
    Computation of the volume of the orientated cell

    :param m_vec_x: normal vector in horizontal direction
    :param m_vec_y: normal vector in vertical direction
    :param spacing_hor: spacing in horizontal direction
    :param spacing_vert: spacing in vertical direction
    :param alpha: plane constant field
    :return:
    """

    m1, m2, dx1, dx2 = order(m_vec_x, m_vec_y, spacing_hor, spacing_vert)
    alpham = m1 * dx1 + m2 * dx2
    area = 0
    if m1==0 and m2==0:
        area = dx1*dx2
    elif m1==0 and m2!=0:
        area = alpha/m2*dx1
    elif m1!=0 and m2!=0:
        if alpha>=0 and alpha<m1*dx1:
            area = alpha**2/(2*m2*m1)
        elif alpha>=m1*dx1 and alpha<alpham-m1*dx1:
            area = alpha*dx1/m2 - m1*dx1**2/(2*m2)
        elif alpha>=alpham-m1*dx1 and alpha<alpham:
            area = dx1*dx2 - (alpha-alpham)**2/(2*m1*m2)
        elif alpha>=alpham:
            area = dx1*dx2
        else:
            area = 0

    return area
def flux_values(flux, m_vec_moment, m_vec_other, alpha, fraction, vel_moment_dir, spacing_moment_dir, spacing_other_dir, dt, dir, cellslope, HN, Nv, Mv):
    """
    Calculation of the flux due updated velocity field. Decission is made to use PLIC or SLIC

    :param m_vec_moment: the normal vector component in direction of velocity field
    :param m_vec_other: the other normal vector component
    :param alpha: plane constant field
    :param fraction: volume fraction field to transport
    :param vel_moment_dir: the corresponding velocity field for flux field
    :param spacing_moment_dir: spacing in the velocity field direction
    :param spacing_other_dir: spacing in other direction
    :param dt: time step
    :param dir: direction corresponding to velocity field; horizontal (0) and vertical (1)
    :param cellslope: slope orientation of the fraction field
    :param HN: Hirt and Nichols (0) or PLIC
    :return:
    """

    # Naming data
    h, dh, do = vel_moment_dir, spacing_moment_dir, spacing_other_dir
    m_vec_abs_moment = np.abs(m_vec_moment)
    m_vec_abs_other = np.abs(m_vec_other)

    for i in range(2, Nv):
        for j in range(2, Mv):
            # Variables
            Ah = h[j, i]
            abs_h = abs(Ah*dt)
            flux[j, i] = 0

            # Acceptor or donor cell
            F_ad, ID1, IA = Fad(j, i, dir, fraction, cellslope, h[j, i])
            fluxPLIC, fluxHN = 0, 0
            if Ah >= 0:
                ID = i * (1 - dir) + dir * j - 1
            else:
                ID = i * (1 - dir) + dir * j

            # You are reconstructed based on F_b^n
            space_mom = spacing_moment_dir[ID] + 0
            space_other = spacing_other_dir[j * (1 - dir) + i * dir] + 0

            # Fraction to prevent zero division
            fract = fraction[ID*dir+j*(1-dir), ID*(1-dir)+i*dir]

            # Determination flux PLIC
            fluxPLICtot = cmpvolume(abs(m_vec_moment[ID * dir + j * (1 - dir), ID * (1 - dir) + i * dir]),
                                    abs(m_vec_other[ID * dir + j * (1 - dir), ID * (1 - dir) + i * dir]), space_mom,
                                    space_other, alpha[ID * dir + j * (1 - dir), ID * (1 - dir) + i * dir])

            if m_vec_abs_moment[ID*dir+j*(1-dir), ID*(1-dir)+i*dir] < 1e-12 and m_vec_abs_other[ID*dir+j*(1-dir), ID*(1-dir)+i*dir] < 1e-12:
                fluxPLIC = fract * abs_h * spacing_other_dir[j * (1 - dir) + i * dir]
            elif m_vec_abs_moment[ID*dir+j*(1-dir), ID*(1-dir)+i*dir] < 1e-12:
                fluxPLIC = min(alpha[ID*dir+j*(1-dir), ID*(1-dir)+i*dir]/(m_vec_abs_other[ID*dir+j*(1-dir), ID*(1-dir)+i*dir]), spacing_other_dir[j*(1-dir)+i*dir]) * abs_h
            elif m_vec_abs_other[ID*dir+j*(1-dir), ID*(1-dir)+i*dir] < 1e-12:
                if np.sign(m_vec_moment[ID*dir+j*(1-dir), ID*(1-dir)+i*dir])*np.sign(h[j, i])<0:
                    fluxPLIC= min(abs_h, fract * spacing_moment_dir[ID]) * spacing_other_dir[j*(1-dir)+i*dir]
                else:
                    fluxPLIC = max(0, abs_h - (1 - fract) * spacing_moment_dir[ID]) * spacing_other_dir[j * (1 - dir) + i * dir]
            elif np.sign(m_vec_moment[ID*dir+j*(1-dir), ID*(1-dir)+i*dir])*np.sign(h[j, i])>0:
                fluxPLIC = cmpvolume(abs(m_vec_moment[ID*dir+j*(1-dir), ID*(1-dir)+i*dir]), abs(m_vec_other[ID*dir+j*(1-dir), ID*(1-dir)+i*dir]), space_mom-abs_h, space_other, alpha[ID*dir+j*(1-dir), ID*(1-dir)+i*dir])
                fluxPLIC = max(0, fluxPLICtot - fluxPLIC)
            elif np.sign(m_vec_moment[ID*dir+j*(1-dir), ID*(1-dir)+i*dir])*np.sign(h[j, i])<0:
                fluxPLIC = cmpvolume(abs(m_vec_moment[ID*dir+j*(1-dir), ID*(1-dir)+i*dir]), abs(m_vec_other[ID*dir+j*(1-dir), ID*(1-dir)+i*dir]), abs_h, space_other, alpha[ID*dir+j*(1-dir), ID*(1-dir)+i*dir])

            fluxPLIC = np.sign(Ah) * min(max(fluxPLIC, 0), fluxPLICtot)

            # Determination flux SLIC
            Ah = h[j, i] + 0
            abs_h = abs(Ah*dt)

            if Ah > 0:
                CF = max((1 - F_ad) * abs_h - (1 - fraction[j - dir, i - 1 + dir]) * dh[
                                    i * (1 - dir) + j * dir - 1], 0)

                fluxHN = np.sign(Ah) * (min((fraction[j - dir, i - 1 + dir] >= (1- 1e-8)) *
                                                (abs_h) + (fraction[j - dir, i - 1 + dir] <
                                                           (1 - 1e-8)) * (F_ad * abs_h + CF),
                                                fraction[j - dir, i - 1 + dir] *
                                                spacing_moment_dir[i * (1 - dir) + j * dir - 1]) *
                                            spacing_other_dir[j*(1-dir)+i*dir])

            elif Ah <= 0:
                CF = max((1 - F_ad) * abs_h - (1 - fraction[j, i])*dh[i*(1-dir) + j*dir], 0)


                fluxHN = np.sign(Ah) * (min((fraction[j, i] >= (1 - 1e-8)) * (abs_h) + (
                                                            fraction[j, i] < (1 - 1e-8)) * (F_ad *
                                                                                                    abs_h + CF),
                                                fraction[j, i] * spacing_moment_dir[i * (1 - dir) + j * dir]) *
                                            spacing_other_dir[j * (1 - dir) + i * dir])

            # Final flux
            flux[j, i] = (HN == 0) * fluxHN + (HN != 0) * fluxPLIC


    return flux
def default_scheme(fraction, m_vec_x, m_vec_y, alpha, vel_hor, vel_vert, spacing_hor, spacing_vert, dt, cellslope, HN):
    """
    No splitting scheme. Fluxing the free surface for same field.

    :param fraction: volume fraction field to transport
    :param m_vec_x: normal vector component in horizontal direction
    :param m_vec_y: normal vector component in vertical direction
    :param alpha: plane constant value
    :param vel_hor: horizontal velocity field
    :param vel_vert: vertical velocity field
    :param spacing_hor: spacing in horizontal direction
    :param spacing_vert: spacing in vertical direction
    :param dt: time step
    :param cellslope: orientation of volume fraction field
    :param HN: Hirt and Nichols (0) or PLIC
    :return:
    """
    # Initial
    Nu, Mu = np.size(fraction, 1) - 1, np.size(fraction, 0) - 2
    Nv, Mv = np.size(fraction, 1) - 2, np.size(fraction, 0) - 1

    flux_x = np.zeros([np.size(fraction, 0), np.size(fraction, 1) + 1])
    flux_y = np.zeros([np.size(fraction, 0) + 1, np.size(fraction, 1)])

    # Flux from edges
    flux_x = flux_values(flux_x, m_vec_x, m_vec_y, alpha, fraction, vel_hor, spacing_hor, spacing_vert, dt, 0, cellslope,
                         HN, Nu, Mu)

    flux_y = flux_values(flux_y, m_vec_y, m_vec_x, alpha, fraction, vel_vert, spacing_vert, spacing_hor, dt, 1,
                         cellslope, HN, Nv, Mv)

    return flux_y, flux_x
def macho_scheme(fraction, m_vec_x, m_vec_y, alpha, vel_hor, vel_vert, spacing_hor, spacing_vert, dt, dir, cellslope, HN):
    """
    Splitting scheme where direction is alternated.

    :param fraction: volume fraction field to transport
    :param m_vec_x: normal vector component in horizontal direction
    :param m_vec_y: normal vector component in vertical direction
    :param alpha: plane constant value
    :param vel_hor: horizontal velocity field
    :param vel_vert: vertical velocity field
    :param spacing_hor: spacing in horizontal direction
    :param spacing_vert: spacing in vertical direction
    :param dt: time step
    :param cellslope: orientation of volume fraction field
    :param HN: Hirt and Nichols (0) or PLIC
    :return:
    """

    # Initial
    Nu, Mu = np.size(fraction, 1) - 1, np.size(fraction, 0) - 2
    Nv, Mv = np.size(fraction, 1) - 2, np.size(fraction, 0) - 1
    N, M = np.size(fraction, 1) - 2, np.size(fraction, 0) - 2

    flux_x = np.zeros([np.size(fraction, 0), np.size(fraction, 1) + 1])
    flux_y = np.zeros([np.size(fraction, 0) + 1, np.size(fraction, 1)])
    Fstar = np.zeros([np.size(fraction, 0), np.size(fraction, 1)])
    eps_f = 1e-10

    cellm = np.chararray((M + 2, N + 2))
    cellm[:] = 'B'
    cellm = cellm.decode("utf-8")

    # dir = determines the y of x direction, when dir = 1 flux_1 in direction of y and flux_2 in x-direction
    if dir == 1:
        vel_1, m_vec_1, spacing_1, flux_1, N1, M1 = vel_vert, m_vec_y, spacing_vert, flux_y, Nv, Mv
        vel_2, m_vec_2, spacing_2, flux_2, N2, M2 = vel_hor, m_vec_x, spacing_hor, flux_x, Nu, Mu
    elif dir == 0:
        vel_1, m_vec_1, spacing_1, flux_1, N1, M1 = vel_hor, m_vec_x, spacing_hor, flux_x, Nu, Mu
        vel_2, m_vec_2, spacing_2, flux_2, N2, M2 = vel_vert, m_vec_y, spacing_vert, flux_y, Nv, Mv

    # All flux values
    flux_1 = flux_values(flux_1, m_vec_1, m_vec_2, alpha, fraction, vel_1, spacing_1, spacing_2, dt, dir, cellslope, HN, N1, M1)

    for i in range(2, N):
         for j in range(2, M):
             Fstar[j, i] = max(fraction[j, i] - (flux_1[j + dir, i + (1-dir)] - flux_1[j, i])/(spacing_1[i*(1 - dir) + j*dir]*spacing_2[i*(dir) + j*(1-dir)])\
                        + dt * max(np.sign(fraction[j, i] - 0.5), 0) * (vel_1[j + dir, i + (1-dir)] - vel_1[j, i])/(spacing_1[i*(1 - dir) + j*dir]), 0)


    # Cell labelling of values
    cellm, placeSm, p_FEs, placeMomm, NOT = \
        cell_labelling_variables(cellm, Fstar, eps_f, N, M)

    # Reconstuction of the free surface
    p_Sm, p_Nm, p_Wm, p_Em, cellslopem, alpham, m_vec_xm, m_vec_ym = \
        reconstruction_variables(placeSm, Fstar, spacing_hor, spacing_vert, HN)

    if dir == 1:
        vel_1, m_vec_1, spacing_1 = vel_vert, m_vec_ym, spacing_vert
        vel_2, m_vec_2, spacing_2 = vel_hor, m_vec_xm, spacing_hor
    elif dir == 0:
        vel_1, m_vec_1, spacing_1 = vel_hor, m_vec_xm, spacing_hor
        vel_2, m_vec_2, spacing_2 = vel_vert, m_vec_ym, spacing_vert

    flux_2 = flux_values(flux_2, m_vec_2, m_vec_1, alpham, Fstar, vel_2, spacing_2, spacing_1, dt, (1-dir), cellslopem, HN, N2, M2)

    if dir == 1:
        flux_x = flux_2
        flux_y = flux_1
    elif dir == 0:
        flux_x = flux_1
        flux_y = flux_2

    return flux_y, flux_x
def cosmic_scheme(fraction, m_vec_x, m_vec_y, alpha, vel_hor, vel_vert, spacing_hor, spacing_vert, dt, cellslope, HN):
    """
    Splitting scheme where direction of transport are averaged
    
    :param fraction: volume fraction field to transport
    :param m_vec_x: normal vector component in horizontal direction
    :param m_vec_y: normal vector component in vertical direction
    :param alpha: plane constant value
    :param vel_hor: horizontal velocity field
    :param vel_vert: vertical velocity field
    :param spacing_hor: spacing in horizontal direction
    :param spacing_vert: spacing in vertical direction
    :param dt: time step
    :param cellslope: orientation of volume fraction field
    :param HN: Hirt and Nichols (0) or PLIC
    :return:
    """

    # Initial
    Nu, Mu = np.size(fraction, 1) - 1, np.size(fraction, 0) - 2
    Nv, Mv = np.size(fraction, 1) - 2, np.size(fraction, 0) - 1
    N, M = np.size(fraction, 1) - 2, np.size(fraction, 0) - 2

    flux_x = np.zeros([np.size(fraction, 0), np.size(fraction, 1) + 1])
    flux_y = np.zeros([np.size(fraction, 0) + 1, np.size(fraction, 1)])
    flux_x2 = np.zeros([np.size(fraction, 0), np.size(fraction, 1) + 1])
    flux_y2 = np.zeros([np.size(fraction, 0) + 1, np.size(fraction, 1)])

    eps_f = 1e-10

    Fx = np.zeros([np.size(fraction, 0), np.size(fraction, 1)])
    Fy = np.zeros([np.size(fraction, 0), np.size(fraction, 1)])

    cellx = np.chararray((M + 2, N + 2))
    celly = np.chararray((M + 2, N + 2))
    cellx[:] = 'B'
    cellx = cellx.decode("utf-8")
    celly[:] = 'B'
    celly = celly.decode("utf-8")

    # First part of flux
    # For dir = 1 flux_1 results in flux_y
    flux_x = flux_values(flux_x, m_vec_x, m_vec_y, alpha, fraction, vel_hor, spacing_hor, spacing_vert, dt, 0,
                         cellslope, HN, Nu, Mu)
    flux_y = flux_values(flux_y, m_vec_y, m_vec_x, alpha, fraction, vel_vert, spacing_vert, spacing_hor, dt, 1,
                         cellslope, HN, Nv, Mv)

    # Update fraction field
    for i in range(2, N):
        for j in range(2, M):
            Fx[j, i] = max(fraction[j, i] - (flux_x[j, i + 1] - flux_x[j, i]) / (
                        spacing_hor[i] * spacing_vert[j]) \
                          + dt * max(np.sign(fraction[j, i] - 0.5), 0) * (
                                      vel_hor[j, i + 1] - vel_hor[j, i]) / (spacing_hor[i]), 0)
            Fy[j, i] = max(fraction[j, i] - (flux_y[j + 1, i] - flux_y[j, i]) / (
                    spacing_hor[i] * spacing_vert[j]) \
                       + dt * max(np.sign(fraction[j, i] - 0.5), 0) * (
                               vel_vert[j + 1, i] - vel_vert[j, i]) / (spacing_vert[j]), 0)

    # Cell labelling of values
    cellx, placeSx, p_FEx, placeMomx, NOT = \
        cell_labelling_variables(cellx, Fx, eps_f, N, M)

    # Reconstuction of the free surface
    p_Sx, p_Nx, p_Wx, p_Ex, cellslopex, alphax, m_vec_xx, m_vec_yx = \
        reconstruction_variables(placeSx, Fx, spacing_hor, spacing_vert, HN)

    flux_y2 = flux_values(flux_y2, m_vec_yx, m_vec_xx, alphax, Fx, vel_vert, spacing_vert, spacing_hor, dt, 1,
                         cellslopex, HN, Nv, Mv)

    # Cell labelling of values
    celly, placeSy, p_FEy, placeMomy, NOT = \
        cell_labelling_variables(celly, Fy, eps_f, N, M)

    # Reconstuction of the free surface
    p_Sy, p_Ny, p_Wy, p_Ey, cellslopey, alphay, m_vec_xy, m_vec_yy = \
        reconstruction_variables(placeSy, Fy, spacing_hor, spacing_vert, HN)

    flux_x2 = flux_values(flux_x2, m_vec_xy, m_vec_yy, alphay, Fy, vel_hor, spacing_hor, spacing_vert, dt, 0,
                         cellslopey, HN, Nu, Mu)

    # Final flux
    flux_x, flux_y = 0.5*(flux_x2+flux_x), 0.5*(flux_y2+flux_y)

    return flux_y, flux_x
def flux_displacement2(spacing_hor, spacing_vert, fraction, cell_height, flux_hor, flux_vert, N, M):
    """
    Fraction update where not height function is applied
    
    :param spacing_hor: spacing in horizontal directioni
    :param spacing_vert: spacing in vertical direction
    :param fraction: volume fraction field
    :param cell_height: field indicated by 1 where HF is applied
    :param flux_hor: horizontal fluxes
    :param flux_vert: vertical fluxes
    :param flux_dif: correction to have pressure equilibrium fluxes for compressible phases
    :return: 
    """
    dx, dy = spacing_hor, spacing_vert

    # Using placeN and length to redistribute as input instead of for loop
    for i in range(2, N):
        for j in range(2, M):
            if (cell_height[j, i]) == 0:
                frac = 1/(dx[i] * dy[j])
                fraction[j, i] = fraction[j, i] + (flux_vert[j, i] + flux_hor[j, i] - flux_vert[j + 1, i] -
                                                           flux_hor[j, i + 1]) *frac

    return fraction
def flux_displacement1(spacing_hor, spacing_vert, fraction, flux_hor, flux_vert, N, M):
    """
    Updating the fraction field in case of no application of height function

    :param spacing_hor: spacing in horizontal direction
    :param spacing_vert: spacing in vertical direction
    :param fraction: volume fraction field
    :param flux_hor: fluxes in horizontal direction
    :param flux_vert: fluxes in vertical direction
    :param flux_dif: fluxes correcting for pressure equilibrium compressible phases
    :return:
    """
    dx, dy = spacing_hor, spacing_vert

    # Using placeN and length to redistribute as input instead of for loop
    for i in range(2, N):
        for j in range(2, M):
            frac = 1 / (dx[i] * dy[j])
            fraction[j, i] = fraction[j, i] + (flux_vert[j, i] + flux_hor[j, i] - flux_vert[j + 1, i] -
                                                       flux_hor[j, i + 1]) * frac

    return fraction
def flux_displacement(spacing_vel_dir, spacing_other_dir, fraction, dir, slope_dir, place, height, flux_vel_dir, flux_other_dir, cell_height2):
    """
    Application of height function update based on the found fluxes

    :param spacing_vel_dir: spacing in direction of the orientation of the height function
    :param spacing_other_dir: spacing in other direction
    :param fraction: fraction field to update
    :param dir: direction of the height function (0) horizontal and (1) vertical
    :param slope_dir: slope direction of the cell (1) N (2) E (3) S (4) W
    :param place: the indices where to apply HF
    :param height: the height value field
    :param flux_vel_dir: the fluxes in direction of the height function
    :param flux_other_dir: the other flux field
    :param flux_dif: the correction to have pressure equilibrium in case of multiple compressible phases
    :param cell: the labelling
    :param Fb1: body fraction field at time level n+1
    :return:
    """
# Dir: y_dir = 1, x_dir = 0
# Slope_dir is the slope
    for x in range(0, (len(place[0, :]) > 1) * len(place[:, 0])):
        j, i = place[x, 0], place[x, 1]
        height[j, i] = height[j, i] + (flux_vel_dir[j - dir, i - 1 + dir] + flux_other_dir[j - dir, i - 1 + dir] -flux_other_dir[j + 1 - 2*dir, i - 1 + 2*dir] +
                                     flux_other_dir[j, i] - flux_other_dir[j + 1 - dir, i + dir] - flux_vel_dir[j + 2*dir, i + 2 - 2*dir] +
                                     flux_other_dir[j + dir, i + 1 - dir] - flux_other_dir[j + 1, i + 1]) / spacing_other_dir[j*(1-dir) + i*dir]

        fraction[int(j + dir*(slope_dir - 2)), int(i + (1 - dir)*(3 - slope_dir))] = min(height[j, i] / spacing_vel_dir[int(i*(1-dir) + j*dir + dir*(slope_dir - 2) + (1 - dir)*(3 - slope_dir))], 1)
        fraction[int(j - dir*(slope_dir - 2)), int(i - (1 - dir)*(3 - slope_dir))] = min(max((height[j, i] - (spacing_vel_dir[int(i*(1-dir) + j*dir)] + spacing_vel_dir[int(i*(1-dir) + j*dir + dir*(slope_dir - 2) + (1 - dir)*(3 - slope_dir))])) / spacing_vel_dir[int(i*(1-dir) + j*dir - dir*(slope_dir - 2) - (1 - dir)*(3 - slope_dir))], 0), 1)
        fraction[j, i] = min(max((height[j, i] - (spacing_vel_dir[int(i*(1-dir) + j*dir + dir*(slope_dir - 2) + (1 - dir)*(3 - slope_dir))])) / spacing_vel_dir[int(i*(1-dir) + j*dir)], 0), 1)
        cell_height2[int(j + dir*(slope_dir - 2)), int(i + (1 - dir)*(3 - slope_dir))] = cell_height2[int(j + dir*(slope_dir - 2)), int(i + (1 - dir)*(3 - slope_dir))] -1
        cell_height2[int(j - dir * (slope_dir - 2)), int(i - (1 - dir) * (3 - slope_dir))] = cell_height2[int(j - dir * (slope_dir - 2)), int(i - (1 - dir) * (3 - slope_dir))] -1
        cell_height2[j, i] = cell_height2[j, i] - 1
    return fraction, cell_height2
def limit_body_value(fraction, N, M):
    for i in range(2, N):
        for j in range(2, M):
            fraction[j, i] = min(max(fraction[j, i], 0), 1)
    return fraction

# Mainfunction
def free_surface_displacement(fraction, cell1, cellslope, vel_hor, vel_vert, spacing_hor, spacing_vert, cell_height, dt, placeN, placeS,
                              placeE, placeW, x_height, y_height,
                              m_vec_x, m_vec_y, alpha, num, HN, HF, MCHS):

    """
    Main function to displace free surface and volume fraction field
    """
    # Domain cells/initial
    N = np.size(fraction, 1) - 2
    M = np.size(fraction, 0) - 2

    cell_height2 = cell_height + 0

    # Step need to be alternating from 1 to 0

    if (num % 2) == 0:
        dir = 0
    else:
        dir = 1

    if MCHS == 1:
        flux_vert, flux_hor = macho_scheme(fraction, m_vec_x, m_vec_y, alpha, vel_hor, vel_vert, spacing_hor, spacing_vert, dt, dir, cellslope, HN)
    elif MCHS == 0:
        flux_vert, flux_hor = cosmic_scheme(fraction, m_vec_x, m_vec_y, alpha, vel_hor, vel_vert, spacing_hor, spacing_vert, dt, cellslope, HN)
    else:
        flux_vert, flux_hor = default_scheme(fraction, m_vec_x, m_vec_y, alpha, vel_hor, vel_vert, spacing_hor, spacing_vert, dt, cellslope, HN)

    # Displacement of free surface
    if HF == 1 or HF == 2:
        fraction, cell_height2 = flux_displacement(spacing_hor, spacing_vert, fraction, 0, 2, placeE, x_height, flux_hor, flux_vert,
                                     cell1, cell_height2)
        fraction, cell_height2 = flux_displacement(spacing_vert, spacing_hor, fraction, 1, 1, placeN, y_height, flux_vert, flux_hor,
                                     cell1, cell_height2)
        fraction, cell_height2 = flux_displacement(spacing_hor, spacing_vert, fraction, 0, 4, placeW, x_height, flux_hor, flux_vert,
                                     cell1, cell_height2)
        fraction, cell_height2 = flux_displacement(spacing_vert, spacing_hor, fraction, 1, 3, placeS, y_height, flux_vert, flux_hor,
                                     cell1, cell_height2)
        fraction = flux_displacement2(spacing_hor, spacing_vert, fraction, cell_height, flux_hor, flux_vert, N, M)

        if np.sum(cell_height2) != 0:
            print('error')

    elif HF == 0:
        fraction = flux_displacement1(spacing_hor, spacing_vert, fraction, flux_hor, flux_vert, N, M)
    else:
        print('Wrong input height function')
        exit()

    # Limit the fraction value between 0, 1
    fraction = limit_body_value(fraction, N, M)


    return fraction, flux_hor, flux_vert, cell_height2
