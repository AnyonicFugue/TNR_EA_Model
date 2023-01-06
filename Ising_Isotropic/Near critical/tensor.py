import matplotlib.pyplot as plt
import scipy
import numba
import numpy as np
import time
n = 6  # Size of tensor network is 2**n * 2**n
h = 0  # External field
# cutoff = 2  # Maximal num of legs. Max num of singular values is 2^cutoff.
ncutoff = 4

j0 = 1  # Interaction constant
no_py = True
para = True
cached = True
upperb = 10**10
lowerb = 10**10

low_kT = 1.8
high_kT = 3
kT_num = 24
sample_num = 1
# Step 0: Initialize the interaction vertices and tensors.
# We first set it to Ising case for simplicity.
# The first two indices mark the tensor it belongs to; the second marks the "edge"
# Last index marks the "spins", 0 for down, 1 for up


# When only 4 tensor is left, direct contract by iteration.
@numba.jit(nopython=no_py, cache=cached)
def direct_contract(T, nx, ny):
    res = 0
    for x0 in range(nx):
        for x1 in range(nx):
            for x2 in range(nx):
                for x3 in range(nx):
                    for y1 in range(ny):
                        for y2 in range(ny):
                            for y3 in range(ny):
                                for y0 in range(ny):
                                    res += T[0][0][x0][y0][x1][y2]*T[0][1][x1][y1][x0][y3] * \
                                        T[1][0][x2][y2][x3][y0] * \
                                        T[1][1][x3][y3][x2][y1]
    return res


@numba.jit(nopython=no_py, cache=cached)
def init(J, T, N0, beta):  # N0 is the initial size of the TN
    tmp = np.zeros(4, dtype=np.int64)
    for i in range(N0):
        for j in range(N0):
            for a in range(2):
                tmp[0] = 2*a-1  # Convert to (\pm 1), as spin
                for b in range(2):
                    tmp[1] = 2*b-1
                    for c in range(2):
                        tmp[2] = 2*c-1
                        for d in range(2):
                            tmp[3] = 2*d-1

                            T[i][j][a][b][c][d] = h*(2*(a+b+c+d)-4)  # Field

                            for k in range(4):
                                T[i][j][a][b][c][d] += tmp[k]*J[i][j][k] * \
                                    (tmp[(k+1) % 4])  # interaction

                            t = T[i][j][a][b][c][d]
                            T[i][j][a][b][c][d] = np.exp(-beta *
                                                         t)

                            if(np.isnan(T[i][j][a][b][c][d]) or np.isinf(T[i][j][a][b][c][d])):
                                print("nan!")

# Step 1: Contraction
# Before contraction, judge whether cutoff is exceeded. If so, use SVD to reduce size.


@numba.jit(nopython=no_py, cache=cached, parallel=para)
def reduce(T, sx, sy, nx, ny):
    tmp_a = np.zeros(((int(sy/2), (nx**2)*(ny), ny)), dtype=np.double)
    tmp_b = np.zeros(((int(sy/2), ny, (nx**2)*(ny))), dtype=np.double)
    # Temporary places for storing the tensors

    u_out = np.zeros(((int(sy/2), nx**2*ny, ny)), dtype=np.double)
    u_out_tr = np.zeros(((int(sy/2), ny, nx**2*ny)), dtype=np.double)
    # return a 1-dim vector,dtype=np.double)
    d_out = np.zeros((int(sy/2), ny), dtype=np.double)
    # return full_dim matrices for compatibility
    v_out = np.zeros(((int(sy/2), ny, ny)), dtype=np.double)

    u = np.zeros(((int(sy/2), (nx**2)*(ny), ny)), dtype=np.double)
    u_tr = np.zeros(((int(sy/2), ny, (nx**2)*(ny))), dtype=np.double)
    d = np.zeros(((int(sy/2), ny)), dtype=np.double)
    d_sorted = np.zeros((int(sy/2), ny), dtype=np.int32)
    v = np.zeros((int(sy/2), ny, ny), dtype=np.double)

    if(np.any(np.isnan(T)) or np.any(np.isinf(T))):
        print("nan!")

    if(ny > ncutoff):
        # perform SVD to reduce number of legs
        for a in numba.prange(int(sy/2)):
            for b in range(sx):
                if(a == 0):
                    y = sy-1
                else:
                    y = 2*a-1

                tmp_a[a] = T[y][b].reshape((nx**2)*(ny), ny)
                tmp_b[a] = np.swapaxes(
                    T[2*a][b], 0, 1).copy().reshape(ny, (nx**2)*(ny))  # change a and b to make b stand in the front.
                # Need to be swapped back when finishing SVD.
                u[a], d[a], v[a] = np.linalg.svd(tmp_a[a], full_matrices=False)

                u_tr[a] = u[a].transpose()
                d_sorted[a] = np.absolute(d[a]).argsort()

                for i in range(ncutoff):
                    u_out_tr[a][i] = u_tr[a][d_sorted[a][ny-1-i]]
                    d_out[a][i] = d[a][d_sorted[a][ny-1-i]]
                    v_out[a][i] = v[a][d_sorted[a][ny-1-i]]

                u_out[a] = u_out_tr[a].transpose()
                T[y][b] = u_out[a].reshape(nx, ny, nx, ny)
                T[2*a][b] = np.swapaxes((np.diag(d_out[a]) @
                                        v_out[a]@tmp_b[a]).reshape(ny, nx, nx, ny), 0, 1)
    if(np.any(np.isnan(T)) or np.any(np.isinf(T))):
        print("nan!")
    return 0


# Here tmp shall also be parallelized, otherwise errors will occur for parallel.
# First write non-parallel, object-mode version; then try to change into parallel and nonpython mode.
@numba.jit(nopython=no_py, cache=cached, parallel=para)
def contract(T, output, sx, sy, nx, ny, flag):
    # Always contract in y direction;
    # sx,sy denote size of TN, while nx,ny denote dimensions in x and y.
    # sx,sy,nx,ny are already exponentiated.
    # ncutoff = 2**cutoff
    # flag denotes whether SVD cutoff is performed
    if(np.any(np.isnan(T)) or np.any(np.isinf(T))):
        print("nan!")

    for a in numba.prange(int(sy/2)):
        for b in range(sx):

            for a1 in range(nx):
                for a2 in range(nx):
                    for c1 in range(nx):
                        for c2 in range(nx):

                            if(flag == False):  # Cutoff isn't performed
                                for b2 in range(ny):
                                    for d1 in range(ny):

                                        output[b][a][b2][a1+nx *
                                                         a2][d1][c1+nx*c2] = 0  # After rotating for 90 degree, the indices need to be rearranged
                                        for s in range(ny):
                                            output[b][a][b2][a1+nx*a2][d1][c1+nx*c2] += T[2 *
                                                                                          a+1][b][a1][s][c1][d1]*T[2*a][b][a2][b2][c2][s]

                            else:  # Cutoff is performed
                                for b2 in range(ncutoff):
                                    for d1 in range(ncutoff):

                                        output[b][a][b2][a1+nx *
                                                         a2][d1][c1+nx*c2] = 0
                                        # This leg isn't affected by SVD
                                        for s in range(ny):
                                            output[b][a][b2][a1+nx*a2][d1][c1+nx*c2] += T[2 *
                                                                                          a+1][b][a1][s][c1][d1]*T[2*a][b][a2][b2][c2][s]
                                        # output[b][a][d1][a1+nx*a2][b2][c1+nx*c2]/=div_num

    if(np.any(np.isnan(output)) or np.any(np.isinf(output))):
        print("nan!")


def calculate(J0, beta):
    nx = 2
    ny = 2
    sx = int(2**n)  # Number of selectable index in x direction
    sy = int(2**n)

    reduce_const = 0  # Records the logarithm of reduction.

    T0 = np.zeros((2**n, 2**n, nx, ny,
                   nx, ny), dtype=np.double)

    init(J0, T0, 2**n, beta)

    T_current = T0  # Storing current data
    T_next = 0  # Storing data after contraction
    # Main loop
    if(np.any(np.isnan(T_current)) or np.any(np.isinf(T_current))):
        print("nan!")

    while(sx > 2 or sy > 2):
        if(ny > ncutoff):
            T_next = np.zeros((int(sx), int(sy/2), ncutoff, nx*nx,
                               ncutoff, nx*nx), dtype=np.double)
            reduce(T_current, sx, sy, nx, ny)
            contract(T_current, T_next, sx, sy, nx, ny, True)
            T_current = T_next

            ny = nx*nx
            nx = ncutoff

            tmp = sy
            sy = sx
            sx = int(tmp/2)

        else:
            T_next = np.zeros((int(sx), int(sy/2), ny, nx*nx,
                               ny, nx*nx), dtype=np.double)
            contract(T_current, T_next, sx, sy, nx, ny, False)
            T_current = T_next

            tmp = ny
            ny = nx*nx
            nx = tmp

            tmp = sy
            sy = sx
            sx = int(tmp/2)

        max_element = max(np.amax(T_current), np.absolute(np.amin(T_current)))

        red_factor = 1.0
        if(max_element > upperb):
            red_factor = max_element/upperb
        # red_factor=np.power(np.amax(T_current),0.5)*div_num
        if(max_element < lowerb):
            red_factor = max_element/lowerb

        if(np.any(np.isnan(T_current)) or np.any(np.isinf(T_current))):
            print("nan!")

        T_current = T_current/red_factor
        reduce_const = reduce_const*2+np.log(red_factor)

    # Here we store the "reduced partition function"

    res_z = direct_contract(T_current, nx, ny)
    # res_q=(np.log(res_z)+2**(2*n)*(np.log(div_num)+adj_num*beta*j0))
    if(res_z <= 0):
        print("nan!")
        return(-1, -1, -1)
    else:
        res_q = np.log(res_z)+reduce_const*4
        res_f = -res_q/beta

    return (res_z, res_q, res_f)


@numba.jit(nopython=no_py, cache=cached)
def anisotropic(J, j1, j2):
    for a in range(2**n):
        for b in range(2**n):
            for k in range(4):
                J[a][b][k] = j1+(k % 2)*(j2-j1)


def main():

    start = time.time()

    x_size = n  # current tensor network size in x direction
    y_size = n  # current TN size in y direction

    F = np.zeros((kT_num, sample_num))
    Z = np.zeros((kT_num, sample_num))
    Q = np.zeros((kT_num, sample_num))
    C = np.zeros((kT_num, sample_num))
    E = np.zeros((kT_num, sample_num))

    kT_array = np.geomspace(low_kT, high_kT, kT_num)

    #beta_array = kT_array
    #beta_array = np.linspace(0.05,10,kT_num)

    step = 0.0001

    for i in range(kT_num):  # Run over all temperatures
        # Repeate several times for spin glasses. Useless for Ising.
        j = 0
        print("currently at "+str(kT_array[i])+'\n')
        for j in range(sample_num):
            print("j="+str(j))
            J0 = np.full((2**n, 2**n, 4), 1)  # Constant coupling

            #anisotropic(J0,1,-1)#initialize as anisotropic coupling

            Z[i][j], Q[i][j], F[i][j] = calculate(J0, 1/kT_array[i])
            z1, q1, f1 = calculate(J0, 1/kT_array[i]+step)
            z2, q2, f2 = calculate(J0, 1/kT_array[i]+2*step)

            E[i][j] = (q1-Q[i][j])/step
            C[i][j] = ((q2-q1)-(q1-Q[i][j]))/(step*step)
    '''
    print(F)
    print(Z)
    print(Q)
    print(C)
    '''

    F_average = np.zeros(kT_num)
    E_average = np.zeros(kT_num)
    C_average = np.zeros(kT_num)

    for i in range(kT_num):
        F_average[i] = np.average(F[i])
        E_average[i] = np.average(E[i])
        C_average[i] = np.average(C[i])

    print(F)
    print(E)
    print("Heat Capacity:")
    print(C)

    end = time.time()

    plt.plot(kT_array,C_average)
    plt.show()

    
    res = open("result.txt", mode='a')
    res.write("Run time="+str(end-start)+'\n')
    res.write("n="+str(n)+",ncutoff="+str(ncutoff)+",kT_num=" +
              str(kT_num)+",sample_num="+str(sample_num)+'\n')
    res.write(str(kT_array)+'\n')
    res.write(str(F_average)+'\n')
    res.write(str(E_average)+'\n')
    res.write(str(C_average)+'\n')
    res.close()
    


if __name__ == "__main__":
    main()
