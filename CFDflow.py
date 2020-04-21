#                                       Import of packages
import time as ti
import numpy as np
import sys

assert sys.version_info >= (3, 6)

#                                       Import of modules
from Modules.Division import safe_div
from Modules import Setup as setup
from Modules import Boundary_condition as bc
from Modules import Variables as variables
from Modules import Intermediate_velocity as starvel
from Modules import Pressure_correction as pressure
from Modules import New_velocity as newvel
from Modules import Free_surface_displacement as fsd
from Modules import Postprocessing as pp
from Modules import Density as den

grid, boundary, x, y, x_node, y_node, z_node, dx, dy, u, v, p, F, domain, t, dt, dt_min, dt_max, iter, CFL, CFLcrit, mu_con_l, \
mu_con_a, rho_con_l, rho_con_a, gamma_a, Heights, Pressures, Mass_data, Save_data, Save_fig, Save_VTK,\
sigma, grav_con, p_0, UPW, AB, HF, GC, HN, COM, MCHS, SLIP, name, atm_pres_side = setup.input()

excelpath, datapath, figurepath, VTKpath = pp.pathmaking(name, Heights, Pressures, Mass_data, Save_data, Save_fig, Save_VTK)

eps_f = CFL[0, 1]
u_old, v_old = np.zeros([np.size(F, 0), np.size(F, 1) + 1]), np.zeros([np.size(F, 0) + 1, np.size(F, 1)])
R_u_old, R_v_old, dt_int, bb = 0, 0, dt, 1
pold = p + 0

# Empty arrays
rho_l = rho_con_l * np.ones([np.size(F, 0), np.size(F, 1)])
rho_a = rho_con_a * np.ones([np.size(F, 0), np.size(F, 1)])

delta_p = pold * 0
F2 = F + 0
N, M = np.size(p, 1) - 2, np.size(p, 0) - 2

# Start calculation
Mtot0, Mtota0, Mtotw0 = pp.mass_conservation(F, dx, dy, rho_l, rho_a)
Mtot, Mtota, Mtotw = Mtot0, Mtota0, Mtotw0
R_u_old, R_v_old, count, time, n, tt = 0, 0, 0, 0, 1, []
H1, H2, H3, H4 = [], [], [], []
P1, P2, P3, P4 = [], [], [], []
Mtot_d, Mtotw_d, Mtota_d = [], [], []

print(' ')
print(' ')
print('Start time calculation')
print('----------------------')

while time < t:
    # print('----------')
    start = ti.time()
    re_calc, pold = False, p + 0

    #   Generating Variables
    C_tilde, rho_cen, mu_cen, cell_height, cell1, cellslope, curv, rho_w, rho_a, rho_l, placeMom,\
    p_s, p_S, p_N, p_E, p_W, p_FE, x_height, y_height, m_vec_x, m_vec_y, alpha, F1 = \
        variables.variables(dx, dy, F, pold, gamma_a, rho_con_a, rho_con_l, p_0, mu_con_a, mu_con_l, eps_f,
                  COM, HN, HF)

    placeSSu, placeSEu, placeMomu = variables.velocity_labelling_variables(0, cell1, np.size(u, 1) - 2, np.size(u, 0) - 2)
    placeSSv, placeSEv, placeMomv = variables.velocity_labelling_variables(1, cell1, np.size(v, 1) - 2, np.size(v, 0) - 2)

    #   Displace free surface algorithm
    F, F_h, F_v, cell_height2 = fsd.free_surface_displacement(F, cell1, cellslope, u, v, dx,
                                                                   dy, cell_height, dt,
                                                                   p_N, p_S, p_E, p_W,
                                                                   x_height, y_height,
                                                                   m_vec_x, m_vec_y, alpha, n,
                                                                   HN, HF, MCHS)

    #   Mass conservation
    Mtot, Mtota, Mtotw = pp.mass_conservation(F1, dx, dy, rho_l, rho_a)

    #   Density at momentum cell faces based on flux
    rho_u, rho_v = den.density_face(rho_cen, rho_l, rho_a, F2, dx, dy, m_vec_x, m_vec_y, alpha, GC)

    #   Intermediate velocity
    u_tilde, R_u = starvel.inter_vel(placeMomu, pold, u, v, dx, dy, 0, rho_u, rho_l, rho_a, F1, cell1, mu_cen, grav_con,
                                     sigma, curv, dt, time, R_u_old, UPW, AB)

    v_tilde, R_v = starvel.inter_vel(placeMomv, pold, v, u, dy, dx, 1, rho_v, rho_l, rho_a, F1, cell1, mu_cen, grav_con,
                                     sigma, curv, dt, time, R_v_old, UPW, AB)

    #   Pressure solver
    delta_p, a_matrix, q_matrix = pressure.poisson_solver(placeMom, pold, u_tilde, v_tilde, dx, dy, rho_a, rho_u, rho_v,
                                                          C_tilde, F1, gamma_a, p_0, dt, atm_pres_side, COM)

    p = delta_p + pold

    #   New velocity
    u_new = newvel.new_vel(placeMomu, u_tilde, dx, dy, delta_p, rho_u, 0, dt)
    v_new = newvel.new_vel(placeMomv, v_tilde, dy, dx, delta_p, rho_v, 1, dt)

    u_new = bc.slip_condition(u_new, 0, SLIP, np.size(u_new, 1) - 2, np.size(u_new, 0) - 2)
    v_new = bc.slip_condition(v_new, 1, SLIP, np.size(v_new, 1) - 2, np.size(v_new, 0) - 2)

    #   CFL controller
    dt_new, count = pp.CFL_controller(u_new, v_new, dx, dy, mu_cen, rho_cen, rho_l, rho_a, sigma, F, CFL, CFLcrit, dt,
                                   dt_min, dt_max, count)

    #   Define variables
    u_old, v_old = u + 0, v + 0
    u, v = u_new + 0, v_new + 0
    F2 = F + 0

    # Old values
    R_u_old, R_v_old = R_u + 0, R_v + 0

    time += dt
    dt = dt_new
    print('----------' )
    print(' ')
    print('     Time:', np.round(time, 6), ' [s]')
    print('     Mass water: ', np.round(safe_div(Mtotw, Mtotw0), 10),
          'Mass air: ', np.round(safe_div(Mtota, Mtota0), 10),
          'Mass total: ', np.round(safe_div(Mtot, Mtot0), 10))

    end = ti.time()
    print('     Cal time', np.round(end - start, 6))
    print('')
    print('')
    print('')

    # Postprocesing
    tt.append(time)
    Mtot_d, Mtotw_d, Mtota_d, P1, P2, P3, P4, H1, H2, H3, H4 = pp.post_processing(Heights, Pressures, Mass_data,
                                                     Save_fig, Save_data, Save_VTK, excelpath, figurepath, datapath,
                                                     VTKpath, tt, domain, F, p,
                                                        x, y, dx, dy, H1, H2, H3, H4, P1, P2, P3, P4, Mtot_d, Mtotw_d, Mtota_d,
                                                        Mtot, Mtot0, Mtotw, Mtotw0, Mtota, Mtota0,
                                                        time, u, v, n, u_tilde, v_tilde, rho_cen,
                                                        cell1, x_node, y_node, z_node)
    n += 1
