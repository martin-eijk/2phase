from shutil import copyfile
import sys
import datetime
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

from Modules.Division import safe_div
from Modules import VTK as vtk

#                                        8. Post-processing
def CFL_controller(hor_vel, vert_vel, hor_spacing, vert_spacing, mu, rho, rho_l, rho_a, coeff, F_b, CFL, CFLcrit,
                   dt, dt_min, dt_max, count):
    # CFL
    CFL_crit = CFL[0, 0]

    # Constraints
    dt_iner_x, dt_iner_y = np.zeros([np.size(F_b, 0), np.size(F_b, 1)]), np.zeros([np.size(F_b, 0), np.size(F_b, 1)])
    dt_visc = np.zeros([np.size(F_b, 0), np.size(F_b, 1)])
    dt_cap = np.zeros([np.size(F_b, 0), np.size(F_b, 1)])

    for i in range(2, np.size(F_b, 1) - 2):
        for j in range(2, np.size(F_b, 0) - 2):
            # Inertial
            dt_iner_y[j, i] = vert_spacing[j] * 0.5 * CFL_crit * abs(
                    safe_div(1, vert_vel[j, i]) - abs(safe_div(1, vert_vel[j, i]))) + vert_spacing[
                                  j - 1] * 0.5 * CFL_crit * abs(
                                      safe_div(1, vert_vel[j, i]) + abs(safe_div(1, vert_vel[j, i])))
            dt_iner_x[j, i] = hor_spacing[i] * 0.5 * CFL_crit * abs(
                    safe_div(1, hor_vel[j, i]) - abs(safe_div(1, hor_vel[j, i]))) + hor_spacing[
                                  i - 1] * 0.5 * CFL_crit * abs(
                                      safe_div(1, hor_vel[j, i]) + abs(safe_div(1, hor_vel[j, i])))

            # Viscous
            dt_visc[j, i] = rho[j, i] * safe_div(1, mu[j, i]) * (hor_spacing[i] * vert_spacing[j]) ** 2 / (
                        2 * hor_spacing[i] ** 2 + 2 * vert_spacing[j] ** 2)

            # Capillary
            dt_cap[j, i] = safe_div(
                (rho_l[j, i] + rho_a[j, i]) * max(vert_spacing[j], hor_spacing[i]) ** 3, 4 * coeff)**0.5

    dt_iner_y[dt_iner_y <= 0] = 1000
    dt_iner_x[dt_iner_x <= 0] = 1000
    dt_visc[dt_visc <= 0] = 1000
    dt_cap[dt_cap <= 0] = 1000

    print('         Timestepping:', dt, np.min(dt_iner_x), np.min(dt_iner_y), np.min(dt_visc), np.min(dt_cap), count)
    if CFLcrit == 1:
        dt_cfl = min(np.min(dt_iner_x), np.min(dt_iner_y))
    else:
        dt_cfl = min(np.min(dt_iner_x), np.min(dt_iner_y), np.min(dt_visc), np.min(dt_cap))

    dt1 = max(min(dt_max, dt_cfl), dt_min)
    # print(count)
    if dt1 < dt:
        dt = 0.5 * dt1
        count = 0
    elif count >= 5 and dt1 >= CFL_crit * dt:
        dt = max(min(dt_max, 1.2 * dt, dt_cfl), dt_min)
        count = 0
    else:
        count += 1

    return dt, count
def mass_conservation(fraction, spacing_hor, spacing_vert, rho_l, rho_a):

    fraction1 = fraction + 0
    Mtotw = np.sum(rho_l[2: np.size(fraction, 0) - 2, 2: np.size(fraction, 1) - 2] * \
                   fraction1[2: np.size(fraction, 0) - 2, 2: np.size(fraction, 1) - 2] *
                   spacing_hor[2: len(spacing_hor) - 2] * spacing_vert[2: len(spacing_vert) - 2, None])
    Mtota = np.sum(rho_a[2: np.size(fraction, 0) - 2, 2: np.size(fraction, 1) - 2] * \
                   (1 - (fraction1[2: np.size(fraction, 0) - 2, 2: np.size(fraction, 1) - 2])) * \
                   spacing_hor[2: len(spacing_hor) - 2] * spacing_vert[2: len(spacing_vert) - 2, None])
    Mtot = Mtota + Mtotw
    return Mtot, Mtota, Mtotw
def pathmaking(name, Heights, Pressures, Mass_data, Save_data, Save_fig, Save_VTK):
    newpath = sys.path[0] + '/Results/' + name + '-' + \
              format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    figurepath = newpath + '/Figures'
    fig_fractionpath = figurepath + '/Fraction'
    fig_velUpath = figurepath + '/VelocityU'
    fig_velVpath = figurepath + '/VelocityV'
    fig_prespath = figurepath + '/Pressure'
    fig_denspath = figurepath + '/Density'
    fig_cellpath = figurepath + '/Cell'
    figurepath = [fig_fractionpath, fig_velUpath, fig_velVpath, fig_denspath, fig_prespath, fig_cellpath]

    VTKpath = newpath + '/VTK'
    VTK_fractionpath = VTKpath + '/VTM/Data/Fraction'
    VTK_velUpath = VTKpath + '/VTM/Data/Uvel'
    VTK_velVpath = VTKpath + '/VTM/Data/Vvel'
    VTK_prespath = VTKpath + '/VTM/Data/Pressure'
    VTK_denspath = VTKpath + '/VTM/Data/Density'
    VTK_cellpath = VTKpath + '/VTM/Data/Cell'
    VTKpath = [VTKpath, VTK_fractionpath, VTK_velUpath, VTK_velVpath, VTK_denspath, VTK_prespath, VTK_cellpath]

    excelpath = newpath + '/Excel'
    heightpath = excelpath + '/Height'
    pressurepath = excelpath + '/Pressure'
    masspath = excelpath + '/Mass_cons'
    excelpath = [heightpath, pressurepath, masspath]

    datapath = newpath + '/Data'
    data_fractionpath = datapath + '/Fraction'
    data_fakefractionpath = datapath + '/fakeFraction'
    data_velUpath = datapath + '/VelocityU'
    data_velVpath = datapath + '/VelocityV'
    data_prespath = datapath + '/Pressure'
    data_denspath = datapath + '/Density'
    data_intervelUpath = datapath + '/InterVelocityU'
    data_intervelVpath = datapath + '/InterVelocityV'
    data_cellpath = datapath + '/Cell'
    datapath = [data_fractionpath, data_velUpath, data_velVpath, data_intervelUpath, data_intervelVpath,
                data_denspath, data_prespath, data_cellpath, data_fakefractionpath]

    if Save_fig[0, 0] == 1 or Save_data[0, 0] == 1 or Heights[0, 0] == 1 or Pressures[0, 0] == 1 or Mass_data[0, 0] == 1 or Save_VTK[0, 0] == 1:
        os.makedirs(newpath)
        copyfile(sys.path[0] + '/input.tex', newpath + '/input_copy.tex')
        copyfile(sys.path[0] + '/CFDflow.py', newpath + '/CFDflow_copy.py')

    if Save_fig[0, 0] == 1:
        if Save_fig[0, 2] == 1:
            os.makedirs(fig_fractionpath)
        if Save_fig[0, 3] == 1:
            os.makedirs(fig_velUpath)
            os.makedirs(fig_velVpath)
        if Save_fig[0, 5] == 1:
            os.makedirs(fig_prespath)
        if Save_fig[0, 4] == 1:
            os.makedirs(fig_denspath)
        if Save_fig[0, 6] == 1:
            os.makedirs(fig_cellpath)

    if Save_VTK[0, 0] == 1:
        if Save_VTK[0, 2] == 1:
            os.makedirs(VTK_fractionpath)
        if Save_VTK[0, 3] == 1:
            os.makedirs(VTK_velUpath)
            os.makedirs(VTK_velVpath)
        if Save_VTK[0, 5] == 1:
            os.makedirs(VTK_prespath)
        if Save_VTK[0, 4] == 1:
            os.makedirs(VTK_denspath)
        if Save_VTK[0, 6] == 1:
            os.makedirs(VTK_cellpath)

    if Heights[0, 0] == 1 or Pressures[0, 0] == 1 or Mass_data[0, 0] == 1:
        if Heights[0, 0] == 1:
            os.makedirs(heightpath)
        if Pressures[0, 0] == 1:
            os.makedirs(pressurepath)
        if Mass_data[0, 0] == 1:
            os.makedirs(masspath)

    if Save_data[0, 0] == 1:
        if Save_data[0, 2] == 1:
            os.makedirs(data_fractionpath)
            os.makedirs(data_fakefractionpath)
        if Save_data[0, 3] == 1:
            os.makedirs(data_velUpath)
            os.makedirs(data_velVpath)
        if Save_data[0, 6] == 1:
            os.makedirs(data_prespath)
        if Save_data[0, 5] == 1:
            os.makedirs(data_denspath)
        if Save_data[0, 7] == 1:
            os.makedirs(data_cellpath)
        if Save_data[0, 4] == 1:
            os.makedirs(data_intervelUpath)
            os.makedirs(data_intervelVpath)

    return excelpath, datapath, figurepath, VTKpath
def heights(Heights, excelpath, tt, domain, fraction, dx, dy, H1, H2, H3, H4):
    Heightxp = np.zeros([np.size(fraction, 0)])
    Heightyp = np.zeros([np.size(fraction, 1)])
    Heightxm = np.zeros([np.size(fraction, 0)])
    Heightym = np.zeros([np.size(fraction, 1)])
    dir1 = int(Heights[0, 2])
    dir2 = int(Heights[0, 4])
    dir3 = int(Heights[0, 6])
    dir4 = int(Heights[0, 8])

    for j in range(2, np.size(fraction, 0) - 2):
        for i in range(np.size(fraction, 1) - 3, 1, -1):
            if fraction[j, i] > 0.01:
                Heightxp[j] = np.sum(dx[2:i+1])-(1-fraction[j, i])*dx[i]
                break

    for i in range(2, np.size(fraction, 1) - 2):
        for j in range(2, np.size(fraction, 0) - 2):
            if fraction[j, i] > 0.01:
                Heightyp[i] = np.sum(dy[j:len(dy)-2])-(1-fraction[j, i])*dy[j]
                break

    for j in range(2, np.size(fraction, 0) - 2):
        for i in range(2, np.size(fraction, 1) - 2):
            if fraction[j, i] > 0.01:
                Heightxm[j] = np.sum(dx[i:len(dx)-2])-(1-fraction[j, i])*dx[i]
                break

    for i in range(2, np.size(fraction, 1) - 2):
        for j in range(np.size(fraction, 0) - 3, 1, -1):
            if fraction[j, i] > 0.01:
                Heightym[i] = np.sum(dy[2:i+1])-(1-fraction[j, i])*dy[j]
                break

    if dir1 == 1:
        pos1 = (0.5 * (Heightyp[int(np.ceil(abs(Heights[0, 1]) / domain[0, 0] * (np.size(fraction, 1) - 4))) + 1] +
                           np.sign(Heights[0, 1]) * Heightyp[
                               int(np.ceil(abs(Heights[0, 1]) / domain[0, 0] * (np.size(fraction, 1) - 4))) + 1]) +
                    0.5 * (Heightym[int(np.ceil(abs(Heights[0, 1]) / domain[0, 0] * (np.size(fraction, 1) - 4))) + 1] -
                           np.sign(Heights[0, 1]) * Heightym[
                               int(np.ceil(abs(Heights[0, 1]) / domain[0, 0] * (np.size(fraction, 1) - 4))) + 1]))
    else:
        pos1 = (0.5 *(Heightxp[int(np.ceil(abs(Heights[0, 1]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1] +
                               np.sign(Heights[0, 1]) * Heightxp[int(np.ceil(abs(Heights[0, 1]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1]) + \
                         0.5 *(Heightxm[int(np.ceil(abs(Heights[0, 1]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1] -
                               np.sign(Heights[0, 1]) * Heightxm[int(np.ceil(abs(Heights[0, 1]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1]))

    if dir2 == 1:
        pos2 = (0.5 * (Heightyp[int(np.ceil(abs(Heights[0, 3]) / domain[0, 0] * (np.size(fraction, 1) - 4))) + 1] +
                           np.sign(Heights[0, 3]) * Heightyp[
                               int(np.ceil(abs(Heights[0, 3]) / domain[0, 0] * (np.size(fraction, 1) - 4))) + 1]) +
                    0.5 * (Heightym[int(np.ceil(abs(Heights[0, 3]) / domain[0, 0] * (np.size(fraction, 1) - 4))) + 1] -
                           np.sign(Heights[0, 3]) * Heightym[
                               int(np.ceil(abs(Heights[0, 3]) / domain[0, 0] * (np.size(fraction, 1) - 4))) + 1]))
    else:
        pos2 = (0.5 *(Heightxp[int(np.ceil(abs(Heights[0, 3]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1] +
                               np.sign(Heights[0, 3]) * Heightxp[int(np.ceil(abs(Heights[0, 3]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1]) + \
                         0.5 *(Heightxm[int(np.ceil(abs(Heights[0, 3]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1] -
                               np.sign(Heights[0, 3]) * Heightxm[int(np.ceil(abs(Heights[0, 3]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1]))

    if dir3 == 1:
        pos3 = (0.5 *(Heightyp[int(np.ceil(abs(Heights[0, 5]) / domain[0, 0] * (np.size(fraction, 1) - 4)))+1] +
                         np.sign(Heights[0, 5])*Heightyp[int(np.ceil(abs(Heights[0, 5]) / domain[0, 0] * (np.size(fraction, 1) - 4)))+1]) +
                   0.5 *(Heightym[int(np.ceil(abs(Heights[0, 5]) / domain[0, 0] * (np.size(fraction, 1) - 4)))+1] -
                         np.sign(Heights[0, 5])*Heightym[int(np.ceil(abs(Heights[0, 5]) / domain[0, 0] * (np.size(fraction, 1) - 4)))+1]))
    else:
        pos3 = (0.5 *(Heightxp[int(np.ceil(abs(Heights[0, 5]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1] +
                               np.sign(Heights[0, 5]) * Heightxp[int(np.ceil(abs(Heights[0, 5]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1]) + \
                         0.5 *(Heightxm[int(np.ceil(abs(Heights[0, 5]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1] -
                               np.sign(Heights[0, 5]) * Heightxm[int(np.ceil(abs(Heights[0, 5]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1]))

    if dir4 == 1:
        pos4 = (0.5 *(Heightyp[int(np.ceil(abs(Heights[0, 7]) / domain[0, 0] * (np.size(fraction, 1) - 4)))+1] +
                         np.sign(Heights[0, 7])*Heightyp[int(np.ceil(abs(Heights[0, 7]) / domain[0, 0] * (np.size(fraction, 1) - 4)))+1]) +
                   0.5 *(Heightym[int(np.ceil(abs(Heights[0, 7]) / domain[0, 0] * (np.size(fraction, 1) - 4)))+1] -
                         np.sign(Heights[0, 7])*Heightym[int(np.ceil(abs(Heights[0, 7]) / domain[0, 0] * (np.size(fraction, 1) - 4)))+1]))
    else:
        pos4 = (0.5 *(Heightxp[int(np.ceil(abs(Heights[0, 7]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1] +
                               np.sign(Heights[0, 7]) * Heightxp[int(np.ceil(abs(Heights[0, 7]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1]) + \
                         0.5 *(Heightxm[int(np.ceil(abs(Heights[0, 7]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1] -
                               np.sign(Heights[0, 7]) * Heightxm[int(np.ceil(abs(Heights[0, 7]) / domain[0, 1] * (np.size(fraction, 0) - 4)))+1]))

    H1.append(pos1)
    H2.append(pos2)
    H3.append(pos3)
    H4.append(pos4)

    outputfilename1 = excelpath[0] + '/Heights_dir' + str(Heights[0, 2]) + '_' + str(Heights[0, 1]) + 'm.csv'
    outputfilename2 = excelpath[0] + '/Heights_dir' + str(Heights[0, 4]) + '_' + str(Heights[0, 3]) + 'm.csv'
    outputfilename3 = excelpath[0] + '/Heights_dir' + str(Heights[0, 6]) + '_' + str(Heights[0, 5]) + 'm.csv'
    outputfilename4 = excelpath[0] + '/Heights_dir' + str(Heights[0, 8]) + '_' + str(Heights[0, 7]) + 'm.csv'

    np.savetxt(outputfilename1, np.c_[tt, H1], delimiter=',', fmt='%10.5f')
    np.savetxt(outputfilename2, np.c_[tt, H2], delimiter=',', fmt='%10.5f')
    np.savetxt(outputfilename3, np.c_[tt, H3], delimiter=',', fmt='%10.5f')
    np.savetxt(outputfilename4, np.c_[tt, H4], delimiter=',', fmt='%10.5f')

    return H1, H2, H3, H4
def pressures(Pressures, excelpath, tt, domain, pressure, P1, P2, P3, P4):

    pos1 = pressure[int(np.ceil((1 - Pressures[0, 2] / domain[0, 1]) * (np.size(pressure, 0) - 4))) + 1,
                    int(np.ceil(Pressures[0, 1] / domain[0, 0] * (np.size(pressure, 1) - 4))) + 1]
    pos2 = pressure[int(np.ceil((1 - Pressures[0, 4] / domain[0, 1]) * (np.size(pressure, 0) - 4))) + 1,
                    int(np.ceil(Pressures[0, 3] / domain[0, 0] * (np.size(pressure, 1) - 4))) + 1]
    pos3 = pressure[int(np.ceil((1 - Pressures[0, 6] / domain[0, 1]) * (np.size(pressure, 0) - 4))) + 1,
                    int(np.ceil(Pressures[0, 5] / domain[0, 0] * (np.size(pressure, 1) - 4))) + 1]
    pos4 = pressure[int(np.ceil((1 - Pressures[0, 8] / domain[0, 1]) * (np.size(pressure, 0) - 4))) + 1,
                    int(np.ceil(Pressures[0, 7] / domain[0, 0] * (np.size(pressure, 1) - 4))) + 1]
    vol = 0
    pres = 0
    # for i in range(2, int(np.round(np.size(fraction, 1) - 2))):
    #     for j in range(int(np.round(np.size(fraction, 0) * 0.5)), int(np.round(np.size(fraction, 0) - 2))):
    #         if fraction[j, i] < 0.95:
    #             vol = vol + (1 - fraction[j, i]) * dx[i] * dy[j]
    #             pres = pres + pressure[j, i] * (1 - fraction[j, i]) * dx[i] * dy[j]

    # pos1 = pres / vol
    P1.append(pos1)
    P2.append(pos2)
    P3.append(pos3)
    P4.append(pos4)

    outputfilename1 = excelpath[1] + '/Pressures_' + str(Pressures[0, 1]) + ',' + str(Pressures[0, 2]) + 'm.csv'
    outputfilename2 = excelpath[1] + '/Pressures_' + str(Pressures[0, 3]) + ',' + str(Pressures[0, 4]) + 'm.csv'
    outputfilename3 = excelpath[1] + '/Pressures_' + str(Pressures[0, 5]) + ',' + str(Pressures[0, 6]) + 'm.csv'
    outputfilename4 = excelpath[1] + '/Pressures_' + str(Pressures[0, 7]) + ',' + str(Pressures[0, 8]) + 'm.csv'

    np.savetxt(outputfilename1, np.c_[tt, P1], delimiter=',', fmt='%10.5f')
    np.savetxt(outputfilename2, np.c_[tt, P2], delimiter=',', fmt='%10.5f')
    np.savetxt(outputfilename3, np.c_[tt, P3], delimiter=',', fmt='%10.5f')
    np.savetxt(outputfilename4, np.c_[tt, P4], delimiter=',', fmt='%10.5f')
    return P1, P2, P3, P4
def figures(Figures, figurepath, time, n, x, y, fraction, u, v, density, pressure, domain, cell):
    alpha = Figures[0, 7]
    if n % Figures[0, 1] == 0:
        if Figures[0, 2] == 1:
            fig1, ax = plt.subplots()
            cmap = plt.get_cmap('Blues', 6)
            cs = ax.contourf(x, np.flip(y), fraction[2:np.size(fraction, 0) - 2, 2:np.size(fraction, 1) - 2], cmap=cmap)
            cmap = plt.get_cmap('binary')
            ax.grid(False)
            ax.set_title('Volume fraction in [-]', fontsize=10)
            ax.set_xlabel('[m]')
            ax.set_ylabel('[m]')
            ax.axis([0, domain[0, 0], 0 , domain[0, 1]])
            fig1.gca().set_aspect('equal', adjustable='box')
            cbar = fig1.colorbar(cs)
            cbar.set_label('[-]', rotation=270)
            plt.savefig(figurepath[0] + '/Fraction_' + str(time) + 's.png')
            plt.close(fig1)
        if Figures[0, 3] == 1:
            fig2, ax = plt.subplots()
            cs = ax.contourf(x, np.flip(y), u[2:np.size(u, 0) - 2, 3:np.size(u, 1) - 2])
            cmap = plt.get_cmap('binary')
            ax.grid(False)
            ax.set_title('Horizontal velocity in [m/s]', fontsize=10)
            ax.set_xlabel('[m]')
            ax.set_ylabel('[m]')
            ax.axis([0, domain[0, 0], 0 , domain[0, 1]])
            fig2.gca().set_aspect('equal', adjustable='box')
            cbar = fig2.colorbar(cs)
            cbar.set_label('[m/s]', rotation=270)
            plt.savefig(figurepath[1] + '/VelocityU_' + str(time) + 's.png')
            plt.close(fig2)
            fig3, ax = plt.subplots()
            cs = ax.contourf(x, np.flip(y), -v[3:np.size(v, 0) - 2, 2:np.size(v, 1) - 2])
            cmap = plt.get_cmap('binary')
            ax.grid(False)
            ax.set_title('Vertical velocity in [m/s]', fontsize=10)
            ax.set_xlabel('[m]')
            ax.set_ylabel('[m]')
            ax.axis([0, domain[0, 0], 0 , domain[0, 1]])
            fig3.gca().set_aspect('equal', adjustable='box')
            cbar = fig3.colorbar(cs)
            cbar.set_label('[m/s]', rotation=270)
            plt.savefig(figurepath[2] + '/VelocityV_' + str(time) + 's.png')
            plt.close(fig3)
        if Figures[0, 4] == 1:
            fig4, ax = plt.subplots()
            cs = ax.contourf(x, np.flip(y), density[2:np.size(fraction, 0) - 2, 2:np.size(fraction, 1) - 2])
            ax.grid(False)
            ax.set_title('Density in [kg/m^3]', fontsize=10)
            ax.set_xlabel('[m]')
            ax.set_ylabel('[m]')
            ax.axis([0, domain[0, 0], 0 , domain[0, 1]])
            fig4.gca().set_aspect('equal', adjustable='box')
            cbar = fig4.colorbar(cs)
            cbar.set_label('[kg/m^3]', rotation=270)
            plt.savefig(figurepath[3] + '/Density_' + str(time) + 's.png')
            plt.close(fig4)
        if Figures[0, 5] == 1:
            fig5, ax = plt.subplots()
            cs = ax.contourf(x, np.flip(y), pressure[2:np.size(fraction, 0) - 2, 2:np.size(fraction, 1) - 2])
            ax.grid(False)
            ax.set_title('Pressure in [N/m^2]', fontsize=10)
            ax.set_xlabel('[m]')
            ax.set_ylabel('[m]')
            ax.axis([0, domain[0, 0], 0 , domain[0, 1]])
            fig5.gca().set_aspect('equal', adjustable='box')
            cbar = fig5.colorbar(cs)
            cbar.set_label('[N/m^2]', rotation=270)
            plt.savefig(figurepath[4] + '/Pressure_' + str(time) + 's.png')
            plt.close(fig5)
        if Figures[0, 6] == 1:
            cell_plot = np.zeros([np.size(cell, 0), np.size(cell, 1)])
            for j in range(2, np.size(cell, 0)-2):
                for i in range(2, np.size(cell, 1)-2):
                    if cell[j, i] == 'F':
                        cell_plot[j, i] = 1
                    if cell[j, i] == 'E':
                        cell_plot[j, i] = 3
                    if cell[j, i] == 'S':
                        cell_plot[j, i] = 2
                    if cell[j, i] == 'C':
                        cell_plot[j, i] = 4
                    if cell[j, i] == 'D':
                        cell_plot[j, i] = 5
                    if cell[j, i] == 'B':
                        cell_plot[j, i] = 6

            cmap = plt.get_cmap('tab10', 8)
            fig6, ax = plt.subplots()
            cs = ax.contourf(x, np.flip(y), cell_plot[2:np.size(cell, 0) - 2, 2:np.size(cell, 1) - 2], cmap=cmap)
            ax.grid(False)
            ax.set_title('Labelling in [-]', fontsize=10)
            ax.set_xlabel('[m]')
            ax.set_ylabel('[m]')
            ax.axis([0, domain[0, 0], 0, domain[0, 1]])
            fig6.gca().set_aspect('equal', adjustable='box')
            cbar = fig6.colorbar(cs)
            cbar.set_label('[-]', rotation=270)
            cs.set_clim(vmin=0, vmax=6)
            plt.savefig(figurepath[5] + '/Cell_' + str(time) + 's.png')
            plt.close(fig6)
def data(Data, datapath, time, n, fraction, u, v, u_tilde, v_tilde, density, pressure, cell):
    if n % Data[0, 1] == 0:
        if Data[0, 2] == 1:
            outputfilename = datapath[0] + '/Fraction_' + str(time) + 's.csv'
            np.savetxt(outputfilename, np.c_[fraction], delimiter=',', fmt='%10.5f')
        if Data[0, 3] == 1:
            outputfilename = datapath[1] + '/U_' + str(time) + 's.csv'
            np.savetxt(outputfilename, np.c_[u], delimiter=',', fmt='%10.5f')
            outputfilename = datapath[2] + '/V_' + str(time) + 's.csv'
            np.savetxt(outputfilename, np.c_[v], delimiter=',', fmt='%10.5f')
        if Data[0, 4] == 1:
            outputfilename = datapath[3] + '/U_tilde_' + str(time) + 's.csv'
            np.savetxt(outputfilename, np.c_[u_tilde], delimiter=',', fmt='%10.5f')
            outputfilename = datapath[4] + '/V_tilde_' + str(time) + 's.csv'
            np.savetxt(outputfilename, np.c_[v_tilde], delimiter=',', fmt='%10.5f')
        if Data[0, 5] == 1:
            outputfilename = datapath[5] + '/Density_' + str(time) + 's.csv'
            np.savetxt(outputfilename, np.c_[density], delimiter=',', fmt='%10.5f')
        if Data[0, 6] == 1:
            outputfilename = datapath[6] + '/Pressure_' + str(time) + 's.csv'
            np.savetxt(outputfilename, np.c_[pressure], delimiter=',', fmt='%10.5f')
        if Data[0, 7] == 1:
            outputfilename = datapath[7] + '/Cell_' + str(time) + 's.csv'
            with open(outputfilename, 'w') as file:
                writer = csv.writer(file, delimiter=',')
                for line in cell:
                    writer.writerow(str(line))
            # np.savetxt(outputfilename, np.c_[cell], delimiter=',')
def mass(tt, excelpath, Mtot_d, Mtotw_d, Mtota_d, Mtot, Mtot0, Mtotw,
         Mtotw0, Mtota, Mtota0):

    Mtot_d.append([safe_div(Mtot, Mtot0)])
    Mtotw_d.append([safe_div(Mtotw, Mtotw0)])
    Mtota_d.append([safe_div(Mtota, Mtota0)])
    outputfilename1 = excelpath[2] + '/Mass.csv'
    np.savetxt(outputfilename1, np.c_[tt, Mtot_d, Mtotw_d, Mtota_d], delimiter=',', fmt='%10.8f')
    return Mtot_d, Mtotw_d, Mtota_d

def post_processing(Heights, Pressures, Mass_data, Save_fig, Save_data, Save_VTK,
                    excelpath, figurepath, datapath, VTKpath, tt, domain, F, p,
                    x, y,
                    dx, dy, H1, H2, H3, H4, P1, P2, P3, P4,
                    Mtot_d, Mtotw_d, Mtota_d, Mtot, Mtot0, Mtotw, Mtotw0, Mtota, Mtota0,
                    time, u, v, n, u_tilde, v_tilde, rho_cen, cell,
                    x_node, y_node, z_node):
    # Height
    if Heights[0, 0] == 1:
        H1, H2, H3, H4 = heights(Heights, excelpath, tt, domain, F, dx, dy, H1, H2, H3, H4)

    # Pressure
    if Pressures[0, 0] == 1:
        P1, P2, P3, P4 = pressures(Pressures, excelpath, tt, domain, p, P1, P2, P3, P4)

    # Mass
    if Mass_data[0, 0] == 1:
        Mtot_d, Mtotw_d, Mtota_d = mass(tt, excelpath, Mtot_d, Mtotw_d, Mtota_d,
                                                Mtot, Mtot0, Mtotw, Mtotw0, Mtota, Mtota0)

    # Figures
    if Save_fig[0, 0] == 1:
        figures(Save_fig, figurepath, time, n, x, y, F, u, v, rho_cen, p, domain, cell)

    # Data
    if Save_data[0, 0] == 1:
        data(Save_data, datapath, time, n, F, u, v, u_tilde, v_tilde, rho_cen, p, cell)

    # VTK
    if Save_VTK[0, 0] == 1:
        vtk.VTKsave(Save_VTK, VTKpath, x_node, y_node, z_node, tt, n, F, u, v, rho_cen, p, cell)

    return Mtot_d, Mtotw_d, Mtota_d, P1, P2, P3, P4, H1, H2, H3, H4