from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy import sparse
import numpy as np
import math as mt

from Modules.Density import cw_density_variables

#                                        4. Pressure solver
def boundary_cond(a_matrix, q_matrix, atm_pres, pres, atm_pres_side, N, M):

    """
    Dirichlet boundary condition multiphase flows, atmospheric pressure
    + undetermined pressures in B-cells
    """

    # Boundary Neumann pressure
    for i in range(2, N):
        a_matrix[(M) * (i - 1), (M) * (i - 1) + 1] = 1
        a_matrix[(M) * i - 1, (M) * i - 2] = 1

    for j in range(2, M):
        a_matrix[j - 1, (M) + j - 1] = 1
        a_matrix[(M) * (N - 1) + j - 1, (M) * (N-2) + j - 1] = 1

    # Boundary Dirichlet pressure atmospheric
    # Boundary Dirichlet pressure atmospheric
    if atm_pres_side == 1 or atm_pres_side == 3:
        for i in range(2, N):
            j = (atm_pres_side == 1) + (atm_pres_side == 3) * M
            n = (M) * (i - 1) + j - 1
            a_matrix[n, :] = 0
            a_matrix[n, n] = -1
            a_matrix[n, n + (atm_pres_side == 1) - (atm_pres_side == 3)] = -1
            q_matrix[n] = -2 * atm_pres + pres[j, i] + pres[j + (atm_pres_side == 1) - (atm_pres_side == 3), i]
    elif atm_pres_side == 2 or atm_pres_side == 4:
        for j in range(2, M):
            i = (atm_pres_side == 4) + (atm_pres_side == 2) * N
            nnn = (M) * (i - 1) + j - 1
            q_matrix[nnn] = -2 * atm_pres + pres[j, i] + pres[j, i + (atm_pres_side == 4) - (atm_pres_side == 2)]
            a_matrix[nnn, :] = 0
            a_matrix[nnn, nnn + M * (atm_pres_side == 4) - M * (atm_pres_side == 2)] = -1

    return a_matrix, q_matrix
def rhs_vector(q_matrix, placeMom, hor_tilde_vel, vert_tilde_vel, spacing_hor, spacing_vert,
               density_air, rho_au, rho_av, C_tilde, fraction, pres, atm_pres, dt, COM, N, M):

    for i in range(2, N):
        for j in range(2, M):
            n = M * (i - 1) + j - 1
            q_matrix[n] = -(atm_pres - pres[j, i])

    for xx in range(0, len(placeMom[:, 0])):
        i, j = placeMom[xx, 1], placeMom[xx, 0]
        n = M * (i - 1) + j - 1
        frac1 = 1 / (C_tilde[j, i] * dt) if C_tilde[j, i] != 0 else 0

        Ahxp = hor_tilde_vel[j, i + 1]
        Ahx = hor_tilde_vel[j, i]
        Ahyp = vert_tilde_vel[j + 1, i]
        Ahy = vert_tilde_vel[j, i]

        if COM == 1:
            terml1 = 0

            terma1 = (1 - fraction[j, i]) * frac1 * (
                    (rho_au[j, i + 1] * Ahxp -
                     rho_au[j, i] * Ahx) / spacing_hor[i] + (rho_av[j + 1, i] * Ahyp -
                                                             rho_av[j, i] * Ahy) / spacing_vert[j] - density_air[
                        j, i] * (
                            (Ahxp - Ahx) / spacing_hor[i] + (Ahyp - Ahy) / spacing_vert[j]))
        else:
            terml1 = 0

            terma1 = 0

        termvel = ((hor_tilde_vel[j, i + 1] - hor_tilde_vel[j, i]) /
                   spacing_hor[i] +
                   (vert_tilde_vel[j + 1, i] - vert_tilde_vel[j, i]) /
                   spacing_vert[
                       j]) / dt


        q_matrix[n] = (termvel + terml1 + terma1)


    return q_matrix
def lhs_matrix(A_matrix, q_matrix, placeMom, spacing_hor, spacing_vert, density_hor, density_vert, fraction,
               density_air, C_tilde, pres, atm_pres, gamma, dt, COM, N, M):

    # Initializing parameters
    dx, dy = spacing_hor, spacing_vert
    rho_u, rho_v = density_hor, density_vert

    for xx in range(0, N*M):
        A_matrix[xx, xx] = -1

    A_matrix[M - 1, N - 1], A_matrix[M * (N - 1), M * (N - 1)] = -1, -1

    for xx in range(0, len(placeMom[:, 0])):
        i, j = placeMom[xx, 1], placeMom[xx, 0]
        ik = i - 1
        jj = j - 1
        n = M * ik + jj

        frac1 = 1 / (dt ** 2 * C_tilde[j, i]) if C_tilde[j, i] != 0 else 0
        frac2 = 1 / (density_air[j, i]) if density_air[j, i] != 0 else 0

        c_a = mt.sqrt(abs(gamma * frac2 * pres[j, i]))

        if COM == 1:
            kk = frac1 * ((1 - fraction[j, i]) / c_a ** 2)
        else:
            kk = 0

        frac1 = 1 / rho_u[j, i + 1] if rho_u[j, i + 1] != 0 else 0
        frac2 = 1 / rho_u[j, i] if rho_u[j, i] != 0 else 0
        frac3 = 1 / rho_v[j + 1, i] if rho_v[j + 1, i] != 0 else 0
        frac4 = 1 / rho_v[j, i] if rho_v[j, i] != 0 else 0

        vol1 = (dx[i + 1] + dx[i])
        vol2 = (dx[i - 1] + dx[i])
        vol3 = (dy[j + 1] + dy[j])
        vol4 = (dy[j - 1] + dy[j])

        A_matrix[n, n + M] = 2 / dx[i] * frac1 * (1 / vol1 if vol1 != 0 else 0)
        A_matrix[n, n - M] = 2 / dx[i] * frac2 * (1 / vol2 if vol2 != 0 else 0)
        A_matrix[n, n] = -(2 / dx[i] * (frac1 * (1 / vol1 if vol1 != 0 else 0) + frac2 * (1 / vol2 if vol2 !=0 else 0)) + 2 / dy[j] *
                           (frac3 * (1 / vol3 if vol3 != 0 else 0) + frac4 * (1 / vol4 if vol4 != 0 else 0))) - \
                         kk
        A_matrix[n, n + 1] = 2 / dy[j] * frac3 * (1 / vol3 if vol3 != 0 else 0)
        A_matrix[n, n - 1] = 2 / dy[j] * frac4 * (1 / vol4 if vol4 != 0 else 0)

        # Boundary condition for robustness
        if frac1 + frac2 + frac3 + frac4 == 0:
            A_matrix[n, :] = 0
            A_matrix[n, n] = -1
            q_matrix[n] = pres[j, i] - atm_pres

    return A_matrix, q_matrix

def poisson_solver(placeMom, pres, vel_tilde_hor, vel_tilde_vert, spacing_hor, spacing_vert, density_air,
                   density_hor, density_vert, C_tilde, fraction, gamma, atm_pres,
                   dt, atm_pres_side, COM):

    # Initialization
    N, M = np.size(fraction, 1) - 2, np.size(fraction, 0) - 2

    a_matrix = np.zeros([(np.size(fraction, 0) - 2) * (np.size(fraction, 1) - 2), (np.size(fraction, 0) - 2) * (np.size(fraction, 1) - 2)])
    q_matrix = np.zeros([(np.size(fraction, 0) - 2) * (np.size(fraction, 1) - 2), 1])

    # Densities
    rho_au = np.zeros([M + 2, N + 3])
    rho_av = np.zeros([M + 3, N + 2])

    rho_au = cw_density_variables(rho_au, density_air, spacing_hor, 0, N+2, M)
    rho_av = cw_density_variables(rho_av, density_air, spacing_vert, 1, M+2, N)

    # RHS of pressure
    q_matrix = rhs_vector(q_matrix, placeMom, vel_tilde_hor, vel_tilde_vert, spacing_hor, spacing_vert,
                          density_air, rho_au, rho_av, C_tilde, fraction, pres, atm_pres, dt, COM, N, M)

    # LHS of pressure
    a_matrix, q_matrix = lhs_matrix(a_matrix, q_matrix, placeMom, spacing_hor, spacing_vert, density_hor, density_vert,
                                    fraction, density_air, C_tilde, pres, atm_pres, gamma, dt, COM, N, M)

    # Boundary conditions
    a_matrix, q_matrix = boundary_cond(a_matrix, q_matrix, atm_pres, pres, atm_pres_side, N, M)
    a_matrix, q_matrix = sparse.csr_matrix(a_matrix), sparse.csr_matrix(q_matrix)

    # Solver
    new_pres = spsolve(a_matrix, q_matrix)

    # Vector to matrix
    new_pres2 = pres + 0
    new_pres2[1:np.size(pres, 0) - 1, 1:np.size(pres, 1) - 1] = new_pres.reshape(np.size(pres, 1) - 2,
                                                                            np.size(pres, 0) - 2).swapaxes(0, 1)

    return new_pres2, a_matrix, q_matrix
