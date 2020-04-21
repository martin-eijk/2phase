import numpy as np

#                                        5. New velocity
def new_vel(placeMom, vel_moment_dir, spacing_moment_dir, spacing_other_dir, pres, density_moment_dir, dir, dt):
    dh, do = spacing_moment_dir, spacing_other_dir
    h = vel_moment_dir
    rho = density_moment_dir
    h_new = vel_moment_dir + 0

    for xx in range(0, (len(placeMom[:, 0]) > 1) * len(placeMom[:, 0])):
        i, j = placeMom[int(xx), 1], placeMom[int(xx), 0]
        V = (dh[i * (1 - dir) + j * dir] + dh[
            i * (1 - dir) + j * dir - 1]) / 2 * do[j * (1 - dir) + i * dir]

        frac = 1 / (V) if V != 0 else 0
        fracrho = 1 / rho[j, i] if rho[j, i] != 0 else 0

        h_new[j, i] = h[j, i] - frac * fracrho * dt * (
                pres[j, i] - pres[j * (1 - dir) + (j - 1) * dir, (1 - dir) * (i - 1) + dir * i]) * do[j * (1 - dir) + i * dir]

    return h_new
