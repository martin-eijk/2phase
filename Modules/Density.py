# Packages
import numpy as np


"""
This module represents the transport of the free surface.
"""


# Subfunctions
def cw_density_variables(rho_face, rho_cen, spacing_moment_dir, dir, Nv, Mv):

    # Density at the cell faces using cell-weighted averaging
    for i in range(1, Nv):
        for j in range(2, Mv):
            term = (spacing_moment_dir[i] + spacing_moment_dir[i - 1])
            frac1 = 1 / term if term != 0 else 0
            rho_face[j * (1 - dir) + i * dir, i * (1 - dir) + j * dir] = (rho_cen[j * (1 - dir) + i * dir, i * (1 - dir) + j * dir] *
                                                                          spacing_moment_dir[i] + rho_cen[
                                                                              j * (1 - dir) + (i - 1) * dir, (i - 1) *
                                                                              (1 - dir) + j * dir] * spacing_moment_dir[
                                                                              i - 1]) * frac1

    return rho_face
def gc_density_variables(rho_face, rho_cen, spacing_moment_dir, spacing_other_dir, alpha, m_vec_moment, m_vec_other, rho_w,
                         rho_a, dir, Nv, Mv):

    # Abs matrices
    m_vec_abs_moment = np.absolute(m_vec_moment)
    m_vec_abs_other = np.absolute(m_vec_other)

    # Gravity consistent interpolation for density at cell faces
    # d_w --> water side
    # d_a --> air side
    for i in range(1, Nv):
        for j in range(2, Mv):
            term1 = m_vec_abs_moment[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)]
            frac1 = 1 / term1 if term1 != 0 else 0

            term = alpha[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)] * frac1 - 0.5 * spacing_moment_dir[i] - \
                   0.5 * m_vec_abs_other[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)] \
                   * frac1 * spacing_other_dir[j]

            d_we = min(max(-np.sign(term) *
                           np.sign(m_vec_moment[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)]) *
                           max(alpha[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)] * frac1
                               - 0.25 * spacing_moment_dir[i] + 0.25 *
                               np.sign(m_vec_moment[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)])
                               * spacing_moment_dir[i] - 0.5 *
                               m_vec_abs_other[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)]
                               * frac1 * spacing_other_dir[j], 0), 0),
                       0.5 * spacing_moment_dir[i]) + abs(
                min(-np.sign(m_vec_moment[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)]) * max(
                    np.sign(term) * 0.5 * spacing_moment_dir[i], 0), 0))
            d_ae = 0.5 * spacing_moment_dir[i] - d_we

            term1 = m_vec_abs_moment[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)]
            frac1 = 1 / term1 if term1 !=0 else 0

            term = alpha[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)] * frac1 - \
                   0.5 * spacing_moment_dir[i - 1] - 0.5 * \
                   m_vec_abs_other[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)] * \
                   frac1 * spacing_other_dir[j]
            d_ww = min(max(
                np.sign(term) * np.sign(m_vec_moment[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)]) *
                max(alpha[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)] * frac1
                    - 0.25 * spacing_moment_dir[i - 1] - 0.25 *
                    np.sign(m_vec_moment[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)])
                    * spacing_moment_dir[i - 1] - 0.5 *
                    m_vec_abs_other[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)]
                    * frac1 * spacing_other_dir[j], 0), 0),
                0.5 * spacing_moment_dir[i - 1]) + abs(
                min(np.sign(m_vec_moment[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)]) * max(
                    np.sign(term) * 0.5 * spacing_moment_dir[i - 1], 0), 0))

            d_aw = 0.5 * spacing_moment_dir[i - 1] - d_ww

            rho_ww = (alpha[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)] == 0) * \
                     rho_cen[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)] + \
                     (alpha[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)] != 0) * \
                     rho_w[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)]
            rho_we = (alpha[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)] == 0) * \
                     rho_cen[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)] + \
                     (alpha[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)] != 0) * \
                     rho_w[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)]
            rho_aw = (alpha[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)] == 0) * \
                     rho_cen[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)] + \
                     (alpha[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)] != 0) * \
                     rho_a[j * (1 - dir) + (i - 1) * dir, j * dir + (i - 1) * (1 - dir)]
            rho_ae = (alpha[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)] == 0) * \
                     rho_cen[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)] + \
                     (alpha[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)] != 0) * \
                     rho_a[j * (1 - dir) + i * dir, j * dir + i * (1 - dir)]

            term1 = 0.5 * spacing_moment_dir[i] + \
                    0.5 * spacing_moment_dir[i - 1]
            rho_face[j * (1 - dir) + i * dir, i * (1 - dir) + j * dir] = (d_aw * rho_aw + d_ae * rho_ae +
                                                                                   d_ww * rho_ww + d_we * rho_we) / term1 if term1 != 0 else 0


    return rho_face

# Mainfunction
def density_face(rho_cen, rho_w, rho_air, fraction, spacing_hor,
                              spacing_vert, m_vec_x, m_vec_y, alpha, GC):
    """
    Main function to calculate all kind of densities
    """
    # Empty values
    N = np.size(fraction, 1) - 2
    M = np.size(fraction, 0) - 2
    rho_u = np.zeros([np.size(fraction, 0), np.size(fraction, 1) + 1])
    rho_v = np.zeros([np.size(fraction, 0) + 1, np.size(fraction, 1)])

    # Density at cell faces
    if GC == 1:
        rho_u = gc_density_variables(rho_u, rho_cen, spacing_hor, spacing_vert, alpha, m_vec_x, m_vec_y,
                                     rho_w, rho_air, 0, N + 2, M)
        rho_v = gc_density_variables(rho_v, rho_cen, spacing_vert, spacing_hor, alpha, m_vec_y, m_vec_x,
                                     rho_w, rho_air, 1, M + 2, N)
    elif GC == 0:
        rho_u = cw_density_variables(rho_u, rho_cen, spacing_hor, 0, N + 2, M)
        rho_v = cw_density_variables(rho_v, rho_cen, spacing_vert, 1, M + 2, N)

    return rho_u, rho_v