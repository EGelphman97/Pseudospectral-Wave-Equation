import numpy as np
import scipy.fftpack as sfft
import scipy.interpolate as interp
import matplotlib.pyplot as plt

PI = np.pi
eps_0 = 8.8541878176e-12 #Electrical permitivitty of free space
u_0 = 12.5663706144e-7 #Magnetic permeability of free space

def spectralLaplacian(Yk, ik1, ik2, c2):
    """
    Function to calculate the Laplacian u = Del*((c^2(x1,x2))Del(y(x1,x2))) in the Fourier domain
    where Del is the two-dimensional gradient operator using Algorithm 7 from the paper "Notes on 
    FFT Based Differentiation" by Stephen Johnson of MIT from May 2011. c^2(x1,x2) represents 
    the square of the wave speed at location (x1,x2)

    Parameters:
        Yk: Two-dimensional Fourier transform Y(k1,k2)
        ik1: i*Meshgrid of wave numbers for dimension 1 
        ik2: i*Meshgrid of wave numbers for dimension 2 
        c2: The function representing wave speed c^2(x1,x2)

    Return: 
        u: Spectral Laplacian u(x1,x2)
    """
    #Step 1 - Calculate Gradient of Yk in Frequency Domain
    gn1 = np.real(sfft.ifft2(ik1*Yk))
    gn2 = np.real(sfft.ifft2(ik2*Yk))
    #Step 2 - Inverse FFT to efficiently calculate c^2(x,y)*Del(Yk) in time domain
    vn1 = np.multiply(c2, gn1)
    vn2 = np.multiply(c2, gn2)
    #Step 3 - Calculate Del dot Del(Yk) in frequency domain to get spectral Laplacian
    Vk1 = sfft.fft2(vn1)
    Vk2 = sfft.fft2(vn2)
    Uk = (ik1*Vk1) + (ik2*Vk2)
    return Uk

def calcEpsxy(xmin, xmax, ymin, ymax, N):
    """
    Function to calculate the elecrical permitivitty $\epsilon$(x,y) over 
    the rectangular domain [0,2pi] x [0,2pi]

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
    #return eps_0*epsxy(xe, ye)
    return np.ones((N,N))

def waveEqSolve2D(N, L, f1):
    """
    Function to solve the two-dimensional wave equation for a scalar version of
    the electric field E(x,y,t) the two-dimensional using the FFT based Pseudo-
    Spectral method with Verlet time stepping

    Parameters:
        N : Square root of 2D FFT Size
        L: Length of One Side of Rectangle
        f1: Fundamental frequency, in Hz

    Return:
        Exyt: Value of x-component of E(x,y,t) field at time t=T_f
    """
    N1 = N2 = N#Assume N1 = N2 in 2D FFT
    h = L/N1
    k_neg = np.arange(-N1//2 + 1,0,1)
    k_pos = np.concatenate((np.arange(N1//2), np.array([0])))
    k1 =  np.concatenate((k_pos, k_neg))
    ik = 1j*k1
    ik1, ik2 = np.meshgrid(ik, ik, indexing='ij')#Each k needs to be a 2d array/matrix/meshgrid
    dt = 2.0*np.sqrt((u_0*eps_0)/(2*(np.max(k1)**2)))*0.1 #dt_max/10

    T_f = 0.01/(PI*f1)
    N_t = int(T_f/dt)
    print("Number of time steps:" + str(N_t))
    xc = h*np.arange(N1) + 0.5*h
    yc = h*np.arange(N2) + 0.5*h
    x, y = np.meshgrid(xc, yc, indexing='ij')
    E_0 = np.exp(-10.0*(np.power(x - PI, 2) + np.power(y - PI, 2)))
    V_0 = np.zeros((N1,N2))

    eps = calcEpsxy(0, L, 0, L, N1)#Calculate eps(x,y)
    c2xy = 1.0/(u_0*eps_0*eps)

    #Set up Verlet Method
    E_oldk = sfft.fft2(E_0)
    E_curk = E_oldk + dt*sfft.fft2(V_0)
    E_newk = E_newk = np.zeros((N1,N2))

    #q2 = ((dt**2)/(u_0*eps_0))*(np.power(k1, 2) + np.power(k2, 2))

    for tt in np.arange(N_t):
        #Verlet
        if tt > 1:
            RHSk = spectralLaplacian(E_curk, ik1, ik2, c2xy)
            E_newk =  2.0*E_curk + (dt**2)*RHSk - E_oldk
            E_oldk = E_curk
            E_curk = E_newk
    
    Exyt = np.real(sfft.ifft2(E_curk))
    return Exyt

def main():
    L = 2.0*PI
    N1 = N2 = 256 #Assume N1 = N2 in 2D FFT
    h = L/N1
    k_neg = np.arange(-N1//2 + 1,0,1)
    k_pos = np.concatenate((np.arange(N1//2), np.array([0])))
    kk = np.concatenate((k_pos, k_neg))
    k1, k2 = np.meshgrid(kk,kk, indexing='ij')
    f1 = 1.0/(4.0*PI*np.sqrt(eps_0*u_0))
    tau = 0.01/(PI*f1)

    #Initial Condition
    xc = h*np.arange(N1) + 0.5*h
    yc = h*np.arange(N2) + 0.5*h
    x, y = np.meshgrid(xc, yc, indexing='ij')
    E_0 = np.exp(-10.0*(np.power(x - PI, 2) + np.power(y - PI, 2)))
    fig, ax = plt.subplots(1,3)
    cs1 = ax[0].contourf(x, y, E_0)
    cbar1 = fig.colorbar(cs1)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title("Initial Condition")

    #Analytic Solution
    lambda_nm = np.sqrt((1.0/(u_0*eps_0))*(np.power(k1, 2) + np.power(k2, 2)))
    A_n_true = (sfft.fft2(E_0))*np.exp(1j*lambda_nm*tau)
    E_true = np.real(sfft.ifft2(A_n_true))
    cs2 = ax[1].contourf(x, y, E_true)
    cbar2 = fig.colorbar(cs2)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title("Analytic Solution")
    
    #Numerical Solution
    En = waveEqSolve2D(N1, L, f1)
    cs3 = ax[2].contourf(x, y, En)
    cbar3 = fig.colorbar(cs3)
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')
    ax[2].set_title("Numerical Solution")

    plt.suptitle('Electric Field E(x,y,Tf)')
    plt.show()

if __name__ == "__main__":
    main()