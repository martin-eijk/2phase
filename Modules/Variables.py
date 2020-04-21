import numpy as np
import math as mt

from Modules.Boundary_condition import slip_condition
from Modules.Boundary_condition import fraction_condition

#                                        2. Defining variables
def fluid_density_variables(rho, pres, gamma, density_con, atm_pres, iso, COM, N, M):
    """
    In this function, the density at the cell center is calculated for air or liquid

    density_con: the initial density rho_0
    pres: pressure field
    atm_pres: atmospheric pressure
    gamma_or_c: gamma when air
    iso: 1 when using EoS for speed of sound, 0 when isotropic
    COM: compressible case number
    """

    # Eq of state to calculate density
    # Iso = 1 --> rho_l
    # Iso = 0 --> rho_a
    if COM == 0:
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                rho[j, i] = (1 - iso) * density_con + \
                            iso * density_con
    elif COM == 1:
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                if atm_pres > pres[j, i]:
                    rho[j, i] = (1-iso) * density_con * (1 + 2/np.pi*mt.atan(np.pi/2*((pres[j, i] - atm_pres)/ atm_pres if atm_pres !=0 else 0))) ** (1/gamma) + \
                                iso * density_con
                else:
                    rho[j, i] = (1-iso)*density_con * (pres[j, i] / atm_pres if atm_pres != 0 else 0) ** (1 / gamma) + \
                                iso * density_con

    return rho
def labelling_variables(cell, rho_cen, rho_w, mu_cen, C_tilde, XX, YY, fraction,
                        pres, rho_a, rho_l, mu_a, mu_l, gamma, eps_f, dx, dy, N, M):

    # Values of exact square translation (with labelling of B)
    cell, placeS, p_FE, placeMom, fraction = \
        cell_labelling_variables(cell, fraction, eps_f, N, M)

    # Cell center values
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Safe division
            frac1 = 1 / rho_a[j, i] if rho_a[j, i] != 0 else 0

            # Cell centered values
            rho_w[j, i] = rho_l[j, i]
            rho_cen[j, i] = fraction[j, i] * rho_w[j, i] + (1 - fraction[j, i]) * rho_a[j, i]
            mu_cen[j, i] = fraction[j, i] * mu_l + (1 - fraction[j, i]) * mu_a
            YY[j, i] = fraction[j - 1, i] * dy[j - 1] + fraction[j, i] * dy[j] + fraction[j + 1, i] * dy[j + 1]
            XX[j, i] = fraction[j, i - 1] * dx[i - 1] + fraction[j, i] * dx[i] + fraction[j, i + 1] * dx[i + 1]

            # For C tilde
            c_a = mt.sqrt(abs(gamma * frac1 * pres[j, i]))
            C_tilde[j, i] = rho_cen[j, i]

    # Slip condition needed for around the boundaries
    rho_w = slip_condition(rho_w, 1, 1, N, M)
    rho_w = slip_condition(rho_w, 0, 1, N, M)
    rho_cen = slip_condition(rho_cen, 1, 1, N, M)
    rho_cen = slip_condition(rho_cen, 0, 1, N, M)
    mu_cen = slip_condition(mu_cen, 1, 1, N, M)
    mu_cen = slip_condition(mu_cen, 0, 1, N, M)

    return C_tilde, cell, rho_cen, rho_w, mu_cen, placeMom, placeS, p_FE, XX, YY, fraction
def cell_labelling_variables(cell, fraction, eps_f, N, M):

    # Labelling of E and F cells
    for i in range(2, N):
        for j in range(2, M):
            if fraction[j, i] < eps_f:
                cell[j, i] = 'E'
                fraction[j, i] = 0
            else:
                cell[j, i] = 'F'

    p_F, placeS, p_FE, placeMom = [], [], [], []

    # Labelling of S cells
    for j in range(2, M):
        for i in range(2, N):
            if cell[j, i] == 'F' and (cell[j, i-1] == 'E' or cell[j-1, i] == 'E' or cell[j+1, i] == 'E' or cell[j, i+1] == 'E'):
                cell[j, i] = 'S'
                placeS.append([j, i])
                placeMom.append([j, i])
            elif cell[j, i] == 'F':
                p_F.append([j, i])
                p_FE.append([j, i])
                placeMom.append([j, i])
            else:
                p_FE.append([j, i])
                placeMom.append([j, i])

    if len(placeS) == 0:
        placeS.append([0, 0])

    placeS = np.asarray(placeS)
    placeMom = np.asarray(placeMom)

    return cell, placeS, p_FE, placeMom, fraction
def reconstruction_variables(placeS, fraction, dx, dy, HN):

    # Cellslope determination
    N, M = np.size(fraction, 1) - 2, np.size(fraction, 0) - 2
    p_S, p_N, p_E, p_W = [], [], [], []
    m_vec_x, m_vec_y = np.zeros([M + 2, N + 2]), np.zeros([M + 2, N + 2])
    alpha = np.zeros([M + 2, N + 2])
    cellslope = np.zeros([M + 2, N + 2]).astype(int)

    # Free surface reconstruction PLIC
    alpha, m_vec_x, m_vec_y = youngs_variables(alpha, m_vec_x, m_vec_y, placeS, fraction, dx, dy, HN, N, M)

    for x in range(0, (len(placeS[:, 0]) > 1) * len(placeS[:, 0])):
        j = placeS[x, 0]
        i = placeS[x, 1]
        if abs(round(m_vec_y[j, i], 8)) > abs(round(m_vec_x[j, i], 8)) and round(m_vec_y[j, i], 8) > 0.0:
            cellslope[j, i] = 1
            p_N.append([j, i])
        elif abs(round(m_vec_y[j, i], 8)) > abs(round(m_vec_x[j, i], 8)) and round(m_vec_y[j, i], 8) <= 0.0:
            cellslope[j, i] = 3
            p_S.append([j, i])
        elif abs(round(m_vec_y[j, i], 8)) <= abs(round(m_vec_x[j, i], 8)) and round(m_vec_x[j, i], 8) > 0.0:
            cellslope[j, i] = 4
            p_W.append([j, i])
        elif abs(round(m_vec_y[j, i], 8)) <= abs(round(m_vec_x[j, i], 8)) and round(m_vec_x[j, i], 8) <= 0.0:
            cellslope[j, i] = 2
            p_E.append([j, i])

    if len(p_S) == 0:
        p_S.append([0, 0])
    if len(p_N) == 0:
        p_N.append([0, 0])
    if len(p_W) == 0:
        p_W.append([0, 0])
    if len(p_E) == 0:
        p_E.append([0, 0])

    p_S, p_N = np.asarray(p_S), np.asarray(p_N)
    p_W, p_E = np.asarray(p_W), np.asarray(p_E)

    return p_S, p_N, p_W, p_E, cellslope, alpha, m_vec_x, m_vec_y
def curvature_variables(curv, x_height, y_height, spacing_x, spacing_y, place, cellslope, N, M):
    for xx in range(0, len(place[:, 0])):
        j, i = place[xx, 0], place[xx, 1]
        slope_dir = cellslope[j, i]
        dir = (slope_dir == 3) + (slope_dir == 1)
        height = (1 - dir) * x_height + dir * y_height
        if dir == 1:
            spacing = spacing_x
        else:
            spacing = spacing_y

        if j * (1 - dir) + i * (dir) == 2:
            Hn = (height[j + 1 - dir, i + dir] - height[j, i]) / (
                    0.5 * (spacing[j * (1 - dir) + i * dir + 1] + spacing[j * (1 - dir) + i * dir]))
            Hs = 0
        elif j * (1 - dir) + i * (dir) == (M * (1 - dir) + N * dir) - 1:
            Hn = 0
            Hs = (height[j, i] - height[j - 1 + dir, i - dir]) / (
                    0.5 * (spacing[j * (1 - dir) + i * dir - 1] + spacing[j * (1 - dir) + i * dir]))
        else:
            Hn = (height[j + 1 - dir, i + dir] - height[j, i]) / (
                    0.5 * (spacing[j * (1 - dir) + i * dir + 1] + spacing[j * (1 - dir) + i * dir]))
            Hs = (height[j, i] - height[j - 1 + dir, i - dir]) / (
                    0.5 * (spacing[j * (1 - dir) + i * dir - 1] + spacing[j * (1 - dir) + i * dir]))

        curv[j, i] = 1 / spacing[j * (1 - dir) + i * dir] * (Hn / (mt.sqrt(1 + Hn ** 2)) - Hs / (mt.sqrt(1 + Hs ** 2)))

    return curv
def s_position_variables(cell, place, p_FE, dir):
    p_height = np.zeros([np.size(place, 0), 9]).astype(int)
    amount_height = np.zeros([3]).astype(int)


    # Only the cell combination F S E results in HF
    for x in range(0, (len(place[:, 0]) > 1) * len(place[:, 0])):
        i, j = place[x, 1], place[x, 0]
        if cell[j, i] == 'S' and (cell[j + dir, i + 1 - dir] == 'E' or cell[j - dir, i - 1 + dir] == 'E') and \
                (cell[j + dir, i + 1 - dir] == 'F' or cell[j - dir, i - 1 + dir] == 'F'):
            p_height[amount_height[0], 0], p_height[amount_height[0], 1] = j, i
            amount_height[0] = amount_height[0] + 1
        else:
            p_FE.append([j, i])

    return p_height, amount_height, p_FE
def flux_height_variables(value, amount, slope_dir, dir, p_height, cell_height, placeFE):
    # Amount gives the amount of S cells in a height function to look for
    place = []
    for x in range(0 + int(((1 - dir) * (slope_dir - 2) / 2 + dir * (3 - slope_dir) / 2) * (value[amount - 1] - 1)),
                   -int((1 - dir) * (slope_dir - 2) / 2 + dir * (3 - slope_dir) / 2) + int(
                       (1 - dir) * (4 - slope_dir) / 2 + dir * (slope_dir - 1) / 2) * value[amount - 1],
                   1 - 2 * int((1 - dir) * (slope_dir - 2) / 2 + dir * (3 - slope_dir) / 2)):
        j, i = p_height[x, 2 * (amount - 1)], p_height[x, 1 + 2 * (amount - 1)]
        if cell_height[j - dir * (slope_dir - 2), i - (1 - dir) * (slope_dir - 3)] != 1 and cell_height[j, i] != 1 and \
                cell_height[j + dir * (slope_dir - 2), i + (1 - dir) * (slope_dir - 3)] != 1:
            cell_height[j + dir * (slope_dir - 2), i + (1 - dir) * (slope_dir - 3)] =+ 1
            cell_height[j, i] =+ 1
            cell_height[j - dir * (slope_dir - 2), i - (1 - dir) * (slope_dir - 3)] =+ 1
            place.append([j, i])
        else:
            placeFE.append([j, i])

    if len(place) == 0:
        place.append([_ for _ in range(1)])

    return cell_height, place, placeFE
def youngs_variables(alpha, m_vec_x, m_vec_y, pos_s, fract, spacing_hor, spacing_vert, HN, N, M):
    fract1 = fraction_condition(fract + 0, 1, 1, N, M)
    fract1 = fraction_condition(fract1, 0, 1, N, M)

    # Determination of the normal vector at free surface
    for x in range(0, len(pos_s[:, 0])):
        i = pos_s[x, 1]
        j = pos_s[x, 0]

        NXne = np.sum(fract1[j:j + 2, i + 1] - fract1[j:j+2, i]) / (spacing_hor[i] + spacing_hor[i + 1])
        NXnw = np.sum(fract1[j:j + 2, i] - fract1[j:j+2, i - 1]) / (spacing_hor[i - 1] + spacing_hor[i])
        NXse = np.sum(fract1[j - 1:j + 1, i + 1] - fract1[j - 1:j + 1, i]) / (spacing_hor[i] + spacing_hor[i + 1])
        NXsw = np.sum(fract1[j - 1:j + 1, i] - fract1[j - 1:j + 1, i - 1]) / (spacing_hor[i - 1] + spacing_hor[i])

        NYne = np.sum(fract1[j + 1, i:i + 2] - fract1[j, i:i + 2]) / (spacing_vert[j] + spacing_vert[j + 1])
        NYnw = np.sum(fract1[j + 1, i - 1:i + 1] - fract1[j, i - 1:i + 1]) / (spacing_vert[j] + spacing_vert[j + 1])
        NYse = np.sum(fract1[j, i:i + 2] - fract1[j - 1, i:i + 2]) / (spacing_vert[j - 1] + spacing_vert[j])
        NYsw = np.sum(fract1[j, i - 1:i + 1] - fract1[j - 1, i - 1:i + 1]) / (spacing_vert[j - 1] + spacing_vert[j])

        Nyoungs = [-.25 * (NYne + NYnw + NYse + NYsw), -.25 * (NXne + NXnw + NXse + NXsw)]

        m_vec_x[j, i] = Nyoungs[1] / mt.sqrt((Nyoungs[0])**2 + (Nyoungs[1])**2) if (Nyoungs[0])**2 + (Nyoungs[1])**2 != 0 else 0
        m_vec_y[j, i] = Nyoungs[0] / mt.sqrt((Nyoungs[0])**2 + (Nyoungs[1])**2) if (Nyoungs[0])**2 + (Nyoungs[1])**2 != 0 else 0


        alpha[j, i] = plane_constant_variables(j, i, m_vec_x, m_vec_y, fract, spacing_hor, spacing_vert, HN)

    return alpha, m_vec_x, m_vec_y
def plane_constant_variables(j, i, m_vec_x, m_vec_y, fract, spacing_hor, spacing_vert, HN):
    # Start of plane constant calculation alpha
    if abs(m_vec_x[j, i] * spacing_hor[i]) < abs(m_vec_y[j, i] * spacing_vert[j]):
        if HN == 0:
            m_vec_x[j, i] = 0
            m_vec_y[j, i] = np.sign(m_vec_y[j, i])
        m21 = abs(m_vec_x[j, i])
        dd1 = spacing_hor[i]

        m22 = abs(m_vec_y[j, i])
        dd2 = spacing_vert[j]
    else:
        if HN == 0:
            m_vec_y[j, i] = 0
            m_vec_x[j, i] = np.sign(m_vec_x[j, i])
        m21 = abs(m_vec_y[j, i])
        dd1 = spacing_vert[j]

        m22 = abs(m_vec_x[j, i])
        dd2 = spacing_hor[i]

    V = fract[j, i] * dd1 * dd2

    frac1 = 1 / max((2 * m22), 1e-10)
    V1 = m21 * dd1 ** 2 * frac1
    V2 = dd1 * dd2 - 0.5 * dd1 ** 2 * m21 * frac1 * 2

    if V <= V1 and V >= 0:
        alpha = mt.sqrt(2 * m21 * m22 * V)
    elif V > V1 and V < V2:
        alpha = V * m22 / dd1 + m21 * dd1 / 2
    elif V >= V2:
        alpha = dd1 * m21 + dd2 * m22 - mt.sqrt(abs(2 * dd1 * dd2 * m21 * m22 - 2 * V * m21 * m22))

    return alpha
def object_variables(p_N, p_S, p_E, p_W, cell, spacing_hor, spacing_vert, p_BFE, cell_height):
    p_height_N, amount_height_N, p_BFE = s_position_variables(cell, spacing_hor, p_N, p_BFE, 1)
    p_height_E, amount_height_E, p_BFE = s_position_variables(cell, spacing_vert, p_E, p_BFE, 0)
    p_height_S, amount_height_S, p_BFE = s_position_variables(cell, spacing_hor, p_S, p_BFE, 1)
    p_height_W, amount_height_W, p_BFE = s_position_variables(cell, spacing_vert, p_W, p_BFE, 0)

    cell_height, p_N, p_BFE = flux_height_variables(amount_height_N, 1, 1, 1, p_height_N, cell_height, p_BFE)
    cell_height, p_S, p_BFE = flux_height_variables(amount_height_S, 1, 3, 1, p_height_S, cell_height, p_BFE)
    cell_height, p_E, p_BFE = flux_height_variables(amount_height_E, 1, 2, 0, p_height_E, cell_height, p_BFE)
    cell_height, p_W, p_BFE = flux_height_variables(amount_height_W, 1, 4, 0, p_height_W, cell_height, p_BFE)

    p_S, p_N = np.asarray(p_S), np.asarray(p_N)
    p_W, p_E = np.asarray(p_W), np.asarray(p_E)

    return cell_height, p_S, p_N, p_W, p_E, p_BFE
def velocity_labelling_variables(dir, cell, Nv, Mv):

    ## Determine the FS, SS and FF cell velocities
    placeSS, placeSE = [], []

    # Momentum velocities place
    placeMom = []

    # A cell C is assumed as S-cell for one-phase
    for i in range(2, Nv):
        for j in range(2, Mv):
            if cell[j, i] == 'F' and cell[j - dir, i - 1 + dir ] == 'F':
                placeMom.append([j, i])
            elif (cell[j, i] == 'F' and cell[j - dir, i - 1 + dir] == 'S') or (cell[j, i] == 'S' and cell[j - dir, i - 1 + dir] == 'F'):
                placeMom.append([j, i])
            elif cell[j, i] == 'S' and cell[j - dir, i - 1 + dir] == 'S':
                placeSS.append([j, i])
                placeMom.append([j, i])
            elif (cell[j, i] == 'E' and cell[j - dir, i - 1 + dir] == 'S') or (cell[j, i] == 'S' and cell[j - dir, i - 1 + dir] == 'E'):
                placeMom.append([j, i])
                placeSE.append([j, i])
            elif cell[j, i] == 'E' and cell[j - dir, i - 1 + dir] == 'E':
                placeMom.append([j, i])

    if len(placeSE) == 0:
        placeSE.append([0, 0])
    if len(placeSS) == 0:
        placeSS.append([0, 0])
    if len(placeMom) == 0:
        placeMom.append([0, 0])


    placeMom = np.asarray(placeMom)
    placeSE = np.asarray(placeSE)
    placeSS = np.asarray(placeSS)


    return placeSS, placeSE, placeMom

# Mainfunction
def variables(spacing_hor, spacing_vert, fract, pres, gamma_a,
              rho_con_a, rho_con_l, atm_pres, mu_con_a, mu_con_l, eps_f, COM, HN, HF):

    # Fluid domain
    N, M = np.size(pres, 1) - 2, np.size(pres, 0) - 2
    rho_a = np.zeros([M + 2, N + 2])
    rho_l = np.zeros([M + 2, N + 2])
    rho_cen = np.zeros([M + 2, N + 2])
    rho_w = np.zeros([M + 2, N + 2])
    mu_cen = np.zeros([M + 2, N + 2])
    cell_height = np.zeros([M + 2, N + 2])
    C_tilde = np.zeros([M + 2, N + 2])
    YY = np.zeros([M + 2, N + 2])
    XX = np.zeros([M + 2, N + 2])
    cell = np.chararray((M + 2, N + 2))
    cell[:] = 'W'
    cell = cell.decode("utf-8")

    # Fluid density
    rho_a = fluid_density_variables(rho_a, pres, gamma_a, rho_con_a, atm_pres, 0, COM, N, M)
    rho_l = fluid_density_variables(rho_l, pres, 0, rho_con_l, atm_pres, 1, COM, N, M)

    # Labelling
    fraction1 = fract + 0

    C_tilde, cell, rho_cen, rho_w, mu_cen, placeMom, placeS, p_FE, x_height, y_height, fraction1 = \
        labelling_variables(cell, rho_cen, rho_w, mu_cen, C_tilde, XX, YY, fraction1,
                            pres, rho_a, rho_l, mu_con_a, mu_con_l, gamma_a, eps_f, spacing_hor,
                            spacing_vert, N, M)

    # Reconstuction of the free surface
    p_S, p_N, p_W, p_E, cellslope, alpha, m_vec_x, m_vec_y = \
        reconstruction_variables(placeS, fraction1, spacing_hor, spacing_vert, HN)

    # Curvature
    curv = np.zeros([np.size(x_height, 0), np.size(x_height, 1)])
    curv = curvature_variables(curv, x_height, y_height, spacing_hor, spacing_vert, placeS, cellslope, N, M)

    # Should be based on cell1 if there need to be a HF applied.
    if HF == 1:
        cell_height, p_S, p_N, p_W, p_E, p_FE = object_variables(p_N, p_S, p_E, p_W, cell, spacing_hor,
                                                                 spacing_vert, p_FE, cell_height)

    # Positions where not to apply HF
    p_FE = np.asarray(p_FE)

    return C_tilde, rho_cen, mu_cen, cell_height, cell, cellslope, curv, rho_w, rho_a, rho_l, \
           placeMom, placeS, p_S, p_N, p_E, p_W, p_FE, x_height, y_height, m_vec_x, m_vec_y, alpha, fraction1
