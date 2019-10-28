#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

if __name__=='__main__':
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else :
        print('Usage: run_exe trajectories_file')
        print('trajectories_file format: v_x v_y v_z o_x o_y o_z')
        sys.exit(0)

    trajData = open(file, "r")
    xo = []
    yo = []
    zo = []
    xc = []
    yc = []
    zc = []
    xcr = []
    ycr = []
    zcr = []
    for line in trajData:
        # v_x v_y v_z o_x o_y o_z
        value = [float(s) for s in line.split()]
        if len(value) == 9:
            xo.append(value[0])
            yo.append(value[1])
            zo.append(value[2])
            xc.append(value[3])
            yc.append(value[4])
            zc.append(value[5])
            xcr.append(value[6])
            ycr.append(value[7])
            zcr.append(value[8])
        else:
            continue

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.plot3D(xo, yo, zo, 'black')
    ax1.plot3D(xc, yc, zc, 'green')
    ax1.plot3D(xcr, ycr, zcr, 'red')
    plt.show()

#    p1, = plt.plot(xo, yo, 'b-')
#    p2, = plt.plot(xc, yc, 'y.')
#    p3, = plt.plot(xcr, ycr, 'r.')
#    plt.legend(handles=[p1, p2, p3], \
#               labels=['Pose Odom', 'Pose Camera', 'Pose Camera Refined'])
#    plt.show()
