import numpy as np
import tensorflow as tf

def MinDistanceIndex(r, im, x, y, z):
    DistMatrix = ((2 * np.max(r)) ** 2) * np.ones((im, im))
    for i in np.arange(im):
        for j in np.arange(im):
            if j == i:
                continue
            DistMatrix[i, j] = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2
    index = np.argmin(DistMatrix, axis = 0)
    return index

def computeMetricVort(Vel, gH, im, f, x, y, index):

    Partial_vx = np.zeros((im, 1)) 
    Partial_uy = np.zeros((im, 1))
    Vorticity = np.zeros((im, 1))
    x = x.reshape(-1, )
    y = y.reshape(-1, )

    for i in np.arange(im):
        Partial_vx[i, 0] = (Vel[i, 1] - Vel[index[i], 1])/(x[i] - x[index[i]] + 1e-12)
        Partial_uy[i, 0] = (Vel[i, 0] - Vel[index[i], 0])/(y[i] - y[index[i]] + 1e-12)

    Vorticity = (f + Partial_vx - Partial_uy)/gH
    vort = Vorticity[0, 0]
    return vort
'''
def computeMetricVort(Vel, gH, im, f, x, y, index):

    Partial_vx = np.zeros((im, 1)) 
    Partial_uy = np.zeros((im, 1))
    Vorticity = np.zeros((im, 1))
    x = x.reshape(-1, )
    y = y.reshape(-1, )

    for i in np.arange(im):
        Partial_vx[i, 0] = (Vel[i, 1] - Vel[index[i], 1])/(x[i,0] - x[index[i],0] + 1e-2)
        Partial_uy[i, 0] = (Vel[i, 0] - Vel[index[i], 0])/(y[i,0] - y[index[i],0] + 1e-2)

    Vorticity = (f + Partial_vx - Partial_uy)/gH
    vort = np.sum(Vorticity)
    return vort

def computeMetricTensorVort(Vel, gH, im, f, x, y, index):

    Vorticity = tf.placeholder(shape = (im, 1), dtype = tf.float64)

    Partial_vx = Vel / (x + 1e-2)
    Partial_uy = Vel / (y + 1e-2)
    Vorticity = (f + Partial_vx - Partial_uy)/gH
    vort = tf.reduce_sum(Vorticity)
    return vort
'''    

def computeMetricTensorVort(Vel, gH, im, f, x, y, index):

    pos = index[0]

    Partial_vx = (Vel[0, 1] - Vel[pos, 1]) / (x[0, 0] - x[pos, 0] + 1e-2)
    Partial_uy = (Vel[0, 0] - Vel[pos, 0]) / (y[0, 0] - y[pos, 0] + 1e-2)
    Vorticity = (f[0, 0] + Partial_vx - Partial_uy)/gH[0, 0]
    vort = Vorticity

    return vort
