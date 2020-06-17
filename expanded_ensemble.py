from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from matplotlib import rc
import numpy as np
import os
import imageio
import natsort


def FES(coef, x):
    y = 0
    for i in range(len(coef)):
        y += coef[i] * x ** (len(coef)-1 - i)
    return y


def sphere_on_FES(a, b, c, r):
    # to make it look like a sphere on the plot, we have to consider the scaling of axis
    theta = np.arange(0, np.pi + 0.01, 0.01)
    phi = np.arange(0, 2 * np.pi + 0.01, 0.01)
    theta, phi = np.meshgrid(theta, phi)
    rx = r * np.sin(theta) * np.cos(phi) + a
    ry = (1/12) * r * np.sin(theta) * np.sin(phi) + b
    rz = (40/12) * r * np.cos(theta) + c + (40/12) * (r)  # delta: random noise

    return rx, ry, rz


if __name__ == '__main__':

    os.mkdir('images_EXE')

    coef_start = [-0.000481992, 0.0133127, -0.102597,
                  -0.0406179, 2.84049, -7.0026, 6.84595]
    coef_mid = [-0.000481992, 0.0133127, -0.102597,
                -0.0406179, 2.84049, -7.5026, 6.84595]
    coef_end = [0.0000879645, -0.00136518, -0.016613,
                0.420782, -2.41395, 2.95784, 7.93313]

    # Construct the 3D free energy surface Z
    x = np.arange(0, 12.1, 0.1)
    y = np.arange(0, 1.1, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = FES(coef_start, X)

    # Modify the slice of lambda = 0.5 and lambda between 0 and 0.5
    Z[5] = FES(coef_mid, x)
    delta_start_mid = (Z[5] - Z[0]) / 5
    for i in np.arange(0, 5, 1):
        Z[i] = Z[0] + delta_start_mid * i - np.random.rand(1, len(Z[i]))

    # Modify the slice of lambda = 1 and lambda between 0.5 and 1
    Z[10] = FES(coef_end, x)
    delta_mid_end = (Z[10] - Z[6]) / 5
    for i in np.arange(6, 10, 1):
        Z[i] = Z[6] + delta_mid_end * i - np.random.rand(1, len(Z[i]))

    # Here we modify the surface to avoid the situation that the system is
    # trapped in another basin at other lambda values, so we make the global
    # minimum at least larger than -2.5. (min of y = 0.1 x^2 - x)
    # Z[Z < -2] += 0.1 * Z[Z < -2] ** 2

    """
    # Add the weights
    g_k = np.zeros(len(y))
    for i in range(len(y)):
        g_k[i] = np.mean((Z[i]))
        Z[i] -= g_k[i]
    """

    # Add this to Z to show the surface before adding the weights. The system should stay at lambda = 0
    #for i in np.arange(1, 11, 1):
    #    Z[i] += 2 * i


    # Parameters setup
    # Here we set N_k (the number of step in k direction) as 1
    xx = 1.44908
    yy = 0
    zz = FES(coef_start, xx)
    n_trial = 5
    N_x = 10               # the number of step in k direction
    N_k = 1                # the number of step in x direction
    x_counter = 0          # a counter for MC steps in x direction
    max_d = 1              # max displacement when sampling in the x direction

    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=1, cmap=plt.get_cmap('viridis'), alpha=0.85, zorder=10)
    """
    
    # Expanded ensemble simulation!
    for i in range(n_trial):
        if x_counter < N_x:
            # Metropolis-Hasting algorithm in the x dirction
            x_counter += 1
            delta_x = (2 * np.random.rand(1) - 1) * \
                max_d  # -1 <= delta_x/max_d <= 1
            x_proposed = xx + delta_x
            # to find z_proposed at lambda state i, we find Z[i][j],
            # where j is the index of a number in x array which is closet to xx
            close_xx = min(x, key=lambda x: abs(x - xx))
            j = list(x).index(close_xx)
            # same lambda state, so yy instead of y_proposed
            z_proposed = Z[int(yy * 10)][j]
            delta_z = z_proposed - zz
            if x_proposed < 0.1 or x_proposed > 12:
                p_acc = 0
            else:
                p_acc = np.exp(-delta_z)   # beta = 1 
            r = np.random.rand(1)
            if r < p_acc:
                xx += delta_x
                zz = FES(coef_start, xx)

            # the data of the sphere on the FES, start from lambdda = 0
            rx, ry, rz = sphere_on_FES(xx, yy, zz, 0.2)

        # Every N_x steps, switch the sampling to the k direction
        elif x_counter == N_x:
            x_counter = 0
            # might have to compare with floating numbers (should be y == 0 originally)
            if yy < 0.0001:
                y_proposed = 0.1
            else:
                if np.random.rand(1) > 0.5:
                    y_proposed = yy + 0.1
                else:
                    y_proposed = yy - 0.1
            # to find z_proposed at lambda state i, we find Z[i][j],
            # where j is the index of a number in x array which is closet to xx
            close_xx = min(x, key=lambda x: abs(x - xx))
            j = list(x).index(close_xx)
            z_proposed = Z[int(y_proposed * 20)][j]
            delta_z = z_proposed - zz
            if y_proposed < 0 or y_proposed > 1:
                p_acc = 0
            else:
                p_acc = 5 * np.exp(-delta_z)   # beta = 5 (to accelerate)
            r = np.random.rand(1)

            if r < p_acc:
                yy = y_proposed
                zz = z_proposed

            # the data of the sphere on the FES, start from lambdda = 0
            rx, ry, rz = sphere_on_FES(xx, yy, zz, 0.2)

        rc('font', **{
            'family': 'sans-serif',
            'sans-serif': ['DejaVu Sans'],
            'size': 10,
            'weight': 'bold',
        })
        # Set the font used for MathJax - more on thiprint(images)
        rc('mathtext', **{'default': 'regular'})
        plt.rc('font', family='serif')

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # plot the free energy surface
        surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=1, cmap=plt.get_cmap(
            'viridis'), alpha=0.85, zorder=10)
        # plot the sphere
        #sphere = ax.plot_surface(rx, ry, rz, color='black', zorder=5)

        ax.view_init(elev=30, azim=-60)
        ax.set_xlabel('Collective variable', fontsize='10', fontweight='bold')
        ax.set_xlim3d(0, 13)
        ax.set_ylabel('Coupling parameter $ \lambda $',
                      fontsize='10', fontweight='bold')
        ax.set_ylim3d(0, 1)
        ax.set_zlabel('Free energy ($ k_{B} T$)',
                      fontsize='10', fontweight='bold')
        ax.set_zlim3d(-10, 25)
        #plt.title('Expanded ensemble', fontsize='12', fontweight='bold')
        plt.tight_layout()

        # Add a color bar which maps values to colors.
        # plt.colorbar(surf, shrink=0.5, aspect=20)
        plt.savefig('./images_EXE/expanded_ensemble_%s.png' % (i))
        plt.close('all')

    images = []
    filenames = natsort.natsorted(os.listdir('./images_EXE'), reverse=False)
    # get the filenames and order them in the ascending order

    for filename in filenames:
        if filename.endswith('.png'):
            file_path = os.path.join('./images_EXE', filename)
            images.append(imageio.imread(file_path))
    imageio.mimsave('./EXE_trapped.gif',
                    images, duration=0.05)
    
    #os.system('rm -r images_EXE')

