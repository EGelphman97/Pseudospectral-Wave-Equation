import numpy as np
import scipy.fftpack as sfft
import scipy.interpolate as interp
import matplotlib.pyplot as plt

PI = np.pi
eps_0 = 8.8541878176e-12 #Electrical permitivitty of free space
u_0 = 12.5663706144e-7 #Magnetic permeability of free space

def calcEpsxy(xmin, xmax, ymin, ymax, N):
    """
    Function to calculate the elecrical permitivitty $\epsilon$(x,y) over 
    the rectangular domain [0,$2\pi$] x [0,$2\pi$]

    Parameters:
        xmin, xmax, ymin, ymax: Minimum and Maximum x- and y- values of domain
        N: Number of points along one side of domain. 2D FFT size is 2N

    Return: Interpolated value at (x[i], y[i]) for 0 <= i < N. Ordering of axes
            is consistent with np.meshgrid(x, y, indexing='ij'), not the default
            ordering='xy' 
    """
    M = int(N/4) #Interpolate over coarser set of points
    x = np.linspace(xmin, xmax, M)
    y = np.linspace(ymin, ymax, M)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    eps = np.ones((M,M))
    for i in np.arange(M):
        for j in np.arange(M):
            if (xx[i][j] >= PI/2.0 and xx[i][j] <= (3.0/2.0)*PI) and (yy[i][j] >= PI/2.0 and yy[i][j] <= (3.0/2.0)*PI):
                eps[i][j] = 3.0
    epsxy = interp.RectBivariateSpline(x, y, eps)
    xe = np.linspace(xmin, xmax, N)
    ye = np.linspace(ymin, ymax, N)
    """
    x1, y1 = np.meshgrid(xe, ye, indexing='ij')
    fig, ax = plt.subplots(layout='constrained')
    cs = ax.contourf(x1, y1, epsxy(xe,ye))
    cbar = fig.colorbar(cs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Value of Eps_r(x,y)')
    plt.show()
    """
    #return eps_0*epsxy(xe, ye)#indexing='ij'
    return eps_0*np.ones((N,N))

def waveEqSolve2D(N, L, f1):
    """
    Function to solve the two-dimensional wave equation for the two-dimensional
    Electric Field vector E in x and y using the Pseudo-Spectral method with
    Verlet time stepping

    Parameters:
        N : Square root of 2D FFT Size
        L: Length of One Side of Rectangle
        f1: Fundamental frequency, in Hz

    Return:
        E_x: Value of x-component of E field at time T_f
        E_y: Value of y-component of E field at time T_f
    """
    N1 = N2 = N#Assume N1 = N2 in 2D FFT
    h = L/N1
    k_neg = np.arange(-N1//2 + 1,0,1)
    k_pos = np.concatenate((np.arange(N1//2), np.array([0])))
    k1 = k2 = np.concatenate((k_pos, k_neg))
    dt = 2.0*np.sqrt((u_0*eps_0)/(2*(np.max(k1)**2)))*0.1 #dt_max/10

    T_f = 0.58/(PI*f1)
    N_t = int(T_f/dt)
    print("Number of time steps:" + str(N_t))
    xc = h*np.arange(N1) + 0.5*h
    yc = h*np.arange(N2) + 0.5*h
    x, y = np.meshgrid(xc, yc, indexing='ij')
    E_0 = np.exp(-10.0*(np.power(x - (7.0/4.0)*PI, 2) + np.power(y - (PI/4.0), 2)))
    V_0x = V_0y = np.zeros((N1,N2))

    eps = calcEpsxy(0, L, 0, L, N1)#Calculate eps(x,y)

    #Set up Verlet Method
    E_x_oldk = E_y_oldk = sfft.fft2(E_0)
    E_x_curk = E_x_oldk + dt*sfft.fft2(V_0x)
    E_y_curk = E_y_oldk + dt*sfft.fft2(V_0y)
    E_x_newk = E_y_newk = np.zeros((N1,N2))

    q2 = ((dt**2)/(u_0*eps_0))*(np.power(k1, 2) + np.power(k2, 2))

    for tt in np.arange(N_t):
        #Verlet
        if tt > 1:
            RHSx = np.multiply((dt**2)/(u_0*eps), np.real(sfft.ifft2(np.multiply(-(np.power(k1,2) + np.power(k2,2)), E_x_curk))))
            RHSxk = sfft.fft2(RHSx)
            RHSy = np.multiply((dt**2)/(u_0*eps), np.real(sfft.ifft2(np.multiply(-(np.power(k1,2) + np.power(k2,2)), E_y_curk))))
            RHSyk = sfft.fft2(RHSy)
            E_x_newk =  2.0*E_x_curk + RHSxk - E_x_oldk
            E_y_newk =  2.0*E_y_curk + RHSyk - E_y_oldk
            E_x_oldk = E_x_curk
            E_y_oldk = E_y_curk
            E_x_curk = E_x_newk
            E_y_curk = E_y_newk
    
    E_x = np.real(sfft.ifftn(E_x_curk))
    E_y = np.real(sfft.ifftn(E_y_curk))
    return E_x, E_y

def main():
    L = 2.0*PI
    N1 = N2 = 256 #Assume N1 = N2 in 2D FFT
    h = L/N1
    k_neg = np.arange(-N1//2 + 1,0,1)
    k_pos = np.concatenate((np.arange(N1//2), np.array([0])))
    k1 = k2 = np.concatenate((k_pos, k_neg))
    f1 = 1.0/(4.0*PI*np.sqrt(eps_0*u_0))
    #dt = 2.0*np.sqrt((u_0*eps_0)/(2*(np.max(k1)**2)))*0.1 #dt_max/10
    tau = 0.58/(PI*f1)

    #Initial Condition
    xc = h*np.arange(N1) + 0.5*h
    yc = h*np.arange(N2) + 0.5*h
    x, y = np.meshgrid(xc, yc, indexing='ij')
    E_0 = np.exp(-10.0*(np.power(x - (7.0/4.0)*PI, 2) + np.power(y - (PI/4.0), 2)))
    fig, ax = plt.subplots(1,3)
    print(y)
    print(x)
    print(E_0)
    ax[0].plot(y[0:], E_0[int((7/8)*N1),:])
    #cs1 = ax[0].contourf(x, y, E_0)
    #cbar1 = fig.colorbar(cs1)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title("Initial Condition")

    #Analytic Solution
    lambda_n = np.sqrt((1.0/(u_0*eps_0))*(np.power(k1, 2) + np.power(k2, 2)))
    A_n_true = (sfft.fft2(E_0))*np.cos(lambda_n*tau)
    E_x_true = E_y_true = np.real(sfft.ifft2(A_n_true))
    E_true = np.sqrt(np.power(E_x_true, 2) + np.power(E_y_true, 2))
    cs2 = ax[1].contourf(x, y, E_true)
    cbar2 = fig.colorbar(cs2)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title("Analytic Solution")
    
    #Numerical Solution
    E_x, E_y = waveEqSolve2D(N1, L, f1)
    E2 = np.sqrt(np.power(E_x, 2) + np.power(E_y,2))
    cs3 = ax[2].contourf(x, y, E2)
    cbar3 = fig.colorbar(cs3)
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')
    ax[2].set_title("Numerical Solution")

    plt.suptitle('2-Norm of Electric Field Vector E at Each (x,y)')
    plt.show()

if __name__ == "__main__":
    main()