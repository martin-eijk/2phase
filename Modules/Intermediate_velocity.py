import numpy as np
import math as mt

#                                        3. Intermediate velocity
def second_order_upwind(m_ch, m_co, m_r, m_rr, m_l, m_ll, m_u, m_uu, m_d, m_dd, h_rr, h_r, h_c, h_l, h_ll, h_uu, h_u, h_d, h_dd):
    F_r = 0.5 * m_r * (h_r + h_c) - 0.25 * m_rr *(h_rr - h_r) + 0.25 * abs(m_rr) * (h_rr - 2 * h_r + h_c) +\
        0.25 * m_ch * (h_c - h_l) - 0.25 * abs(m_ch) * (h_r - 2 * h_c + h_l)

    F_l = 0.5 * m_l * (h_l + h_c) - 0.25 * m_ch *(h_r - h_c) + 0.25 * abs(m_ch) * (h_r - 2 * h_c + h_l) +\
        0.25 * m_ll * (h_l - h_ll) - 0.25 * abs(m_ll) * (h_c - 2 * h_l + h_ll)

    F_n = 0.5 * m_u * (h_u + h_c) - 0.25 * m_uu *(h_uu - h_u) + 0.25 * abs(m_uu) * (h_uu - 2 * h_u + h_c) +\
        0.25 * m_co * (h_c - h_d) - 0.25 * abs(m_co) * (h_u - 2 * h_c + h_d)

    F_d = 0.5 * m_d * (h_d + h_c) - 0.25 * m_co * (h_u - h_c) + 0.25 * abs(m_co) * (h_u - 2 * h_c + h_d) + \
          0.25 * m_dd * (h_d - h_dd) - 0.25 * abs(m_dd) * (h_c - 2 * h_d + h_dd)

    return F_r, F_l, F_n, F_d
def first_order_upwind(m_r, m_l, m_u, m_d, h_r, h_c, h_l, h_u, h_d):
    F_r = 0.5 * m_r * (h_r + h_c) - 0.5 * abs(m_r) * h_r + 0.5 * abs(m_r) * h_c

    F_l = 0.5 * m_l * (h_l + h_c) - 0.5 * abs(m_l) * h_c + 0.5 * abs(m_l) * h_l

    F_n = 0.5 * m_u * (h_u + h_c) - 0.5 * abs(m_u) * h_u + 0.5 * abs(m_u) * h_c

    F_d = 0.5 * m_d * (h_d + h_c) - 0.5 * abs(m_d) * h_c + 0.5 * abs(m_d) * h_d

    return F_r, F_l, F_n, F_d
def volume(fracV, V, place, dh, do, dir):
    for xx in range(0, len(place)):
        i, j = place[int(xx), 1], place[int(xx), 0]
        V[j, i] = (dh[i * (1 - dir) + j * dir] + dh[i * (1 - dir) + j * dir - 1]) / 2 * do[j * (1 - dir) + i * dir]
        fracV[j, i] = 1 / V[j, i] if V[j, i] !=0 else 0

    return fracV, V
def nonconservative(j, i, h, o, dir):
    # Check
    h_c = h[j, i]
    o_ls = o[j - dir, i - 1 + dir]
    o_ln = o[j + 1 - 2 * dir, i - 1 + 2 * dir]
    o_rs = o[j, i]
    o_rn = o[j + 1 - dir, i + dir]
    o_rss = o[j - 1 + dir, i - dir]
    o_lss = o[j - 1, i - 1]
    o_rnn = o[j + 2 - 2 * dir, i + 2 * dir]
    o_lnn = o[j + 2 - 3 * dir, i - 1 + 3 * dir]
    h_r = h[j + dir, i + 1 - dir]
    h_l = h[j - dir, i - 1 + dir]

    return h_c, o_ls, o_ln, o_rs, o_rn, o_rss, o_lss, o_rnn, o_lnn, h_r, h_l
def convective(conv, convo, convh, place, h, o, dh, do, upwind, dir):
    # This function calculates the convective term
    # It is possible to apply it for a 2D Cartesian mesh
    # - Moment direction is indicated with [h]
    # - Other direction is indicated with [o]
    # - When upwind=1, second-order upwind, when =0, firs-order upwind

    for xx in range(0, (len(place[:, 0]) > 1)*len(place[:, 0])):
        i, j = place[int(xx), 1], place[int(xx), 0]

        # Mass flux
        h_c, o_ls, o_ln, o_rs, o_rn, o_rss, o_lss, o_rnn, o_lnn, h_r, h_l = nonconservative(j, i, h, o, dir)

        m_ch = h_c * do[j * (1 - dir) + i * dir]
        m_co = 1 / 2 * (1 / 2 * (o_ls + o_ln) * dh[(i - 1) * (1 - dir) + (j - 1) * dir] +
                        1 / 2 * (o_rs + o_rn) * dh[i * (1 - dir) + j * dir])
        m_r = 1 / 2 * (h_r + h_c) * do[j * (1 - dir) + i * dir]
        m_rr = h_r * do[j * (1 - dir) + i * dir]
        m_l = 1 / 2 * (h_l + h_c) * do[j * (1 - dir) + i * dir]
        m_ll = h_l * do[j * (1 - dir) + i * dir]

        m_d = 1 / 2 * (o_rs * dh[i * (1 - dir) + j * dir] + o_ls * dh[(i - 1) * (1 - dir) + (j - 1) * dir])
        m_dd = 1 / 2 * (1 / 2 * (o_rs * dh[i * (1 - dir) + j * dir] + o_ls * dh[(i - 1) * (1 - dir) + (j - 1) * dir]) +
                        1 / 2 * (o_rss * dh[(i) * (1 - dir) + (j) * dir] + o_lss * dh[
                    (i - 1) * (1 - dir) + (j - 1) * dir]))
        m_u = 1 / 2 * (o_rn * dh[i * (1 - dir) + j * dir] + o_ln * dh[(i - 1) * (1 - dir) + (j - 1) * dir])
        m_uu = 1 / 2 * (1 / 2 * (o_rnn * dh[i * (1 - dir) + j * dir] + o_lnn * dh[(i - 1) * (1 - dir) + (j - 1) * dir]) +
                        1 / 2 * (o_rn * dh[i * (1 - dir) + j * dir] + o_ln * dh[(i - 1) * (1 - dir) + (j - 1) * dir]))

        # Velocities
        h_rr = h[j + 2 * dir, i + 2 - 2 * dir]
        h_r = h[j + dir, i + 1 - dir]
        h_c = h[j, i]
        h_l = h[j - dir, i - 1 + dir]
        h_ll = h[j - 2 * dir, i - 2 + 2 * dir]
        h_d = h[j - 1 + dir, i - dir]
        h_dd = h[j - 2 + 2 * dir, i - 2 * dir]
        h_u = h[j + 1 - dir, i + dir]
        h_uu = h[j + 2 - 2 * dir, i + 2 * dir]

        # Choice of scheme
        F_r2, F_l2, F_u2, F_d2 = second_order_upwind(m_ch, m_co, m_r, m_rr, m_l, m_ll, m_u, m_uu, m_d, m_dd, h_rr, h_r, h_c, h_l, h_ll, h_uu, h_u, h_d, h_dd)
        F_r1, F_l1, F_u1, F_d1 = first_order_upwind(m_r, m_l, m_u, m_d, h_r, h_c, h_l, h_u, h_d)

        if upwind == 1:
            F_r = F_r2
            F_l = F_l2
            F_d = F_d2
            F_u = F_u2
        else:
            F_r = F_r1
            F_l = F_l1
            F_d = F_d1
            F_u = F_u1

        convh[j, i] = F_r - F_l
        convo[j, i] = F_u - F_d

        # Conservative er not
        conv[j, i] = convo[j, i] + convh[j, i] - h_c * (m_r-m_l+m_u-m_d)

    return conv
def diffusive(diff, place, h, o, dh, do, mu_cen, rho, dir):
    # Diffusion

    for xx in range(0, (len(place[:, 0]) > 1)*len(place[:, 0])):
        i, j = place[int(xx), 1], place[int(xx), 0]

        # Values
        # Density is on the same position as the velocity!
        # Direction of momentum; x=0, y=1

        # Distance
        dhc = 0.5 * (dh[i * (1 - dir) + j * dir - 1] + dh[i * (1 - dir) + j * dir])
        dos = 0.5 * do[j * (1 - dir) + i * dir] + do[j * (1 - dir) + i * dir + 1] * 0.5
        don = 0.5 * do[j * (1 - dir) + i * dir] + do[j * (1 - dir) + i * dir - 1] * 0.5

        # Viscosities
        frac = do[j * (1 - dir) + i * dir - 1] * mu_cen[j, i] + do[j * (1 - dir) + i * dir] * mu_cen[
            j - 1 + dir, i - dir]
        mune = mu_cen[j, i] * mu_cen[j - 1 + dir, i - dir] * (
                    do[j * (1 - dir) + i * dir - 1] + do[j * (1 - dir) + i * dir]) / frac if frac !=0 else 0
        frac = do[j * (1 - dir) + i * dir + 1] * mu_cen[j, i] + do[j * (1 - dir) + i * dir] * mu_cen[
            j + 1 - dir, i + dir]
        muse = mu_cen[j, i] * mu_cen[j + 1 - dir, i + dir] * (
                    do[j * (1 - dir) + i * dir + 1] + do[j * (1 - dir) + i * dir]) / frac if frac !=0 else 0
        frac = do[j * (1 - dir) + i * dir - 1] * mu_cen[j - dir, i - 1 + dir] + do[j * (1 - dir) + i * dir] \
               * mu_cen[j - 1, i - 1]
        munw = mu_cen[j - dir, i - 1 + dir] * mu_cen[j - 1, i - 1] * (
                    do[j * (1 - dir) + i * dir - 1] + do[j * (1 - dir) + i * dir]) / frac if frac !=0 else 0
        frac = do[j * (1 - dir) + i * dir + 1] * mu_cen[j - dir, i - 1 + dir] + do[j * (1 - dir) + i * dir] \
               * mu_cen[j + 1 - 2 * dir, i - 1 + 2 * dir]
        musw = mu_cen[j - dir, i - 1 + dir] * mu_cen[j + 1 - 2 * dir, i - 1 + 2 * dir] * (
                    do[j * (1 - dir) + i * dir + 1] + do[j * (1 - dir) + i * dir]) / frac if frac !=0 else 0

        mud = (musw * dh[i * (1 - dir) + j * dir - 1] + muse * dh[i * (1 - dir) + j * dir]) / (
                dh[i * (1 - dir) + j * dir] + dh[i * (1 - dir) + j * dir - 1])
        muu = (munw * dh[i * (1 - dir) + j * dir - 1] + mune * dh[i * (1 - dir) + j * dir]) / (
                dh[i * (1 - dir) + j * dir] + dh[i * (1 - dir) + j * dir - 1])

        # Term
        frac1 = 1 / rho[j, i] if rho[j, i] != 0 else 0
        diff[j, i] = frac1 * ((4 / 3 * mu_cen[j, i] * h[j + dir, i + 1 - dir] / (dhc * dh[i * (1 - dir) + j * dir]) +
                         4 / 3 * mu_cen[j - dir, i - 1 + dir] * h[j - dir, i - 1 + dir] / (
                                     dhc * dh[i * (1 - dir) + j * dir - 1]) +
                         mud * h[j + 1 - dir, i + dir] / (do[j * (1 - dir) + i * dir] * dos) +
                         muu * h[j - 1 + dir, i - dir] / (do[j * (1 - dir) + i * dir] * don) -
                         (4 / 3 * mu_cen[j - dir, i - 1 + dir] / (dhc * dh[i * (1 - dir) + j * dir - 1]) + 4 / 3 * mu_cen[
                             j, i] / (
                                  dhc * dh[i * (1 - dir) + j * dir]) +
                          mud / (do[j * (1 - dir) + i * dir] * don) + muu / (
                                  do[j * (1 - dir) + i * dir] * don)) * h[j, i]) + (-
                                                                                    (mud * o[
                                                                                        j + 1 - 2 * dir, i - 1 + 2 * dir] + muu *
                                                                                     o[j, i]) / (
                                                                                            do[j * (
                                                                                                        1 - dir) + i * dir] * dhc) + 1 / (
                                                                                            do[j * (
                                                                                                        1 - dir) + i * dir] * dhc) * (
                                                                                            mud * o[
                                                                                        j + 1 - dir, i + dir] + muu * o[
                                                                                                j - dir, i - 1 + dir])) -
                        2 / 3 * 1 / (do[j * (1 - dir) + i * dir] * dhc) * (
                                    mu_cen[j, i] * (o[j + 1 - dir, i + dir] - o[j, i]) +
                                    mu_cen[j - dir, i - 1 + dir] *
                                    (o[j - dir, i - 1 + dir] - o[j + 1 - 2 * dir, i - 1 + 2 * dir])))

    return diff
def gravity(grav, place, dh, do, grav_con, dir):

    for xx in range(0, (len(place[:, 0]) > 1)*len(place[:, 0])):
        i, j = place[int(xx), 1], place[int(xx), 0]
        grav[j, i] = -do[j * (1 - dir) + i * dir] * grav_con[0, 0 + dir] * (
                0.5 * dh[i * (1 - dir) + j * dir - 1] + 0.5 * dh[i * (1 - dir) + j * dir])

    return grav
def capillary(tens, place, rho_w, rho_a, do, dir, cell, fraction, coeff, curv):
    # For the slope North=1, East=2, South=3, West=4
    for xx in range(0, (len(place[:, 0]) > 1)*len(place[:, 0])):
        i, j = place[int(xx), 1], place[int(xx), 0]
        tens0 = 0

        if (cell[j - dir, i - 1 + dir] == 'S') and (cell[j, i] == 'S'):
            tens0 = do[j * (1 - dir) + i * dir] * (curv[j - dir, i - 1 + dir] + curv[j, i]) / 2 * coeff * (
                    fraction[j - dir, i - 1 + dir] - fraction[j, i])
        elif (cell[j, i] == 'S'):
            tens0 = do[j * (1 - dir) + i * dir] * curv[j, i] * coeff * (
                    fraction[j - dir, i - 1 + dir] - fraction[j, i])
        elif (cell[j - dir, i - 1 + dir] == 'S'):
            tens0 = do[j * (1 - dir) + i * dir] * curv[j - dir, i - 1 + dir] * coeff * (
                    fraction[j - dir, i - 1 + dir] - fraction[j, i])

        tens[j, i] = 2 / (rho_w[j, i] + rho_a[j, i]) * tens0
    return tens
def pressure(P, place, rho, do, pres, dir):
    for xx in range(0, (len(place[:, 0]) > 1)*len(place[:, 0])):
        i, j = place[int(xx), 1], place[int(xx), 0]
        fracrho = 1 / rho[j, i] if rho[j, i] != 0 else 0
        P[j, i] = fracrho * (pres[j, i] - pres[j * (1 - dir) + (j - 1) * dir, (1 - dir) * (i - 1) + dir * i]) * do[j * (1 - dir) + i * dir]

    return P

# Mainfunction
def inter_vel(placeMom, p_old, vel_moment_dir, vel_other_dir, spacing_moment_dir, spacing_other_dir, dir,
              density_moment_dir, rho_w, rho_a, fraction, cell, mu_cen, grav_con, coeff, curv, dt, t, rhs_moment_dir_old, upwind, AB):

    # Empty spaces
    diff, conv = np.zeros([np.size(vel_moment_dir, 0), np.size(vel_moment_dir, 1)]), \
                 np.zeros([np.size(vel_moment_dir, 0), np.size(vel_moment_dir, 1)])
    convh, convo = np.zeros([np.size(vel_moment_dir, 0), np.size(vel_moment_dir, 1)]), \
                   np.zeros([np.size(vel_moment_dir, 0), np.size(vel_moment_dir, 1)])
    tens, grav = np.zeros([np.size(vel_moment_dir, 0), np.size(vel_moment_dir, 1)]), \
                 np.zeros([np.size(vel_moment_dir, 0), np.size(vel_moment_dir, 1)])
    pres = np.zeros([np.size(vel_moment_dir, 0), np.size(vel_moment_dir, 1)])
    fracV, V = np.zeros([np.size(vel_moment_dir, 0), np.size(vel_moment_dir, 1)]), \
               np.zeros([np.size(vel_moment_dir, 0), np.size(vel_moment_dir, 1)])

    # Naming
    dh, do = spacing_moment_dir, spacing_other_dir
    h, o = vel_moment_dir, vel_other_dir
    rho = density_moment_dir

    # Calculation of matrices
    fracV, V = volume(fracV, V, placeMom, dh, do, dir)
    conv = convective(conv, convo, convh, placeMom, h, o, dh, do, upwind, dir)
    diff = diffusive(diff, placeMom, h, o, dh, do, mu_cen, rho, dir)
    grav = gravity(grav, placeMom, dh, do, grav_con, dir)
    tens = capillary(tens, placeMom, rho_w, rho_a, do, dir, cell, fraction, coeff, curv)
    pres = pressure(pres, placeMom, rho, do, p_old, dir)

    R = -conv * fracV + diff + grav * fracV + tens * fracV

    if t == 0 or AB == 0:
        vel_tilde = h + dt * R - dt * fracV * pres
    else:
        vel_tilde = h + 1.5 * dt * R - dt * fracV * pres - 0.5 * dt * rhs_moment_dir_old

    return vel_tilde, R
