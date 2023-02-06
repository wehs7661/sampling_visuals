import copy 
import argparse
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rc
np.random.seed(0)

def initialize():
    parser = argparse.ArgumentParser(
        description='This code demonstrates two possible cases of free energy surface that can fail expanded ensemble.')
    parser.add_argument('-f',
                        '--font',
                        choices=['Arial', 'Serif'],
                        default='Arial',
                        help='The font for the figures.')
    args_parse = parser.parse_args()
    return args_parse

def FES(coef, x):
    y = 0
    for i in range(len(coef)):
        y += coef[i] * x ** (len(coef)-1 - i)
    return y

if __name__ == "__main__":
    args = initialize()
    rc('font', **{
        'family': 'sans-serif',
        'sans-serif': ['DejaVu Sans'],
        'size': 10,
        'weight': 'bold',
    })
    # Set the font used for MathJax - more on thiprint(images)
    rc('mathtext', **{'default': 'regular'})
    plt.rc('font', family=args.font)

    # A template surface to start with
    coef_start = [-0.000481992, 0.0133127, -0.102597,
                  -0.0406179, 2.84049, -7.0026, 6.84595]
    coef_mid = [-0.000481992, 0.0133127, -0.102597,
                -0.0406179, 2.84049, -7.5026, 6.84595]

    x = np.arange(0, 12.1, 0.1)
    y = np.arange(0, 1.1, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = FES(coef_start, X)

    # Scenario A: Deep free energy basins in the alchemical space
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    coef_end = [0.0000879645, -0.00136518, -0.016613,
                0.420782, -2.41395, 2.95784, 7.93313]

    Z1 = copy.deepcopy(Z)
    Z1[5] = FES(coef_mid, x)
    delta_start_mid = (Z1[5] - Z1[0]) / 5
    for i in np.arange(0, 5, 1):
        Z1[i] = Z1[0] + delta_start_mid * i - np.random.rand(1, len(Z1[i]))
    Z1[10] = FES(coef_end, x)
    delta_mid_end = (Z1[10] - Z1[6]) / 5
    for i in np.arange(6, 10, 1):
        Z1[i] = Z1[6] + delta_mid_end * i - np.random.rand(1, len(Z1[i]))
    Z1 -= np.min(Z1)

    surf = ax1.plot_surface(X, Y, Z1, rstride=2, cstride=1, cmap=plt.get_cmap(
            'viridis'), alpha=0.9, zorder=10)
    ax1.view_init(elev=30, azim=-60)
    ax1.set_xlabel('Alchemical variable', fontsize='10', fontweight='bold')
    ax1.set_xlim3d(0, 13)
    ax1.set_ylabel('Alchemical variable',
                    fontsize='10', fontweight='bold')
    ax1.set_ylim3d(0, 1)
    ax1.set_zlim3d(0, 40)
    ax1.text(-0.5, 0, 60, 'Scenario A', weight='bold', fontsize=16)
    ax1.set_zlabel('Free energy ($ kT$)',
                    fontsize='10', fontweight='bold')          
    plt.tight_layout()
    plt.savefig('Scenario_A.png', dpi=600)

    # Scenario B: A free energy barrier present for all lambda states
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')
    coef_end = [-0.000481992, 0.0133127, -0.102597,
                  -0.0406179, 2.84049, -7.5026, 6.84595]  # we keep other coefs the same

    Z2 = copy.deepcopy(Z)   # below we modify the free energy surface from the template
    Z2[5] = FES(coef_mid, x)
    delta_start_mid = (Z2[5] - Z2[0]) / 5
    for i in np.arange(0, 5, 1):
        Z2[i] = Z2[0] + delta_start_mid * i - 2 * np.random.rand(1, len(Z2[i]))
    Z2[10] = FES(coef_end, x)
    delta_mid_end = (Z2[10] - Z2[6]) / 5
    for i in np.arange(1, 10, 1):
        Z2[i] = Z2[1] + delta_mid_end * i - (6 * np.random.rand(1, len(Z2[i])) - 3)
    Z2 -= np.min(Z2)

    surf = ax2.plot_surface(X, Y, Z2, rstride=2, cstride=1, cmap=plt.get_cmap(
            'viridis'), alpha=0.9, zorder=10)
    ax2.view_init(elev=30, azim=-60)
    ax2.set_xlabel('Collective variable', fontsize='10', fontweight='bold')
    ax2.set_xlim3d(0, 13)
    ax2.set_ylabel('Alchemical variable',
                    fontsize='10', fontweight='bold')
    ax2.set_ylim3d(0, 1)
    ax2.set_zlim3d(0, 40)
    ax2.text(-0.5, 0, 60, 'Scenario B', weight='bold', fontsize=16)
    ax2.set_zlabel('Free energy ($ kT$)',
                    fontsize='10', fontweight='bold')
    
    
    plt.tight_layout()
    plt.savefig('Scenario_B.png', dpi=600)