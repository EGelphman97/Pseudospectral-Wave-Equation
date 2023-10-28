#Eric Gelphman
#Colorado School of Mines Department of Applied Mathematics and Statistics
#Ahmed Ahmed
#Colorado School of Mines Department of Geophysics
#Test script to verify program works with variable c(x,y) and source term
#October 27, 2023

import numpy as np
import scipy.fftpack as sfft
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

def waveEqSolve2D(N, L, Tf):
    """
    Function to solve the two-dimensional wave equation for a scalar version of
    the electric field E(x,y,t) the two-dimensional using the FFT based Pseudo-
    Spectral method with Verlet time stepping

    Parameters:
        N : Square root of 2D FFT Size
        L: Length of One Side of Rectangle
        Tf: Final time value of numerical solution, in seconds        

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
    omega = 1000

    N_t = int(Tf/dt + 0.5)#Number of time steps
    print("Number of time steps:" + str(N_t))
    xc = np.linspace(0, L, num=N1)
    x, y = np.meshgrid(xc, xc, indexing='ij')

    #Calculate the position-dependent wave velocity c(x,y) and source term f(x,y,t)
    cxy = np.multiply(np.cos(x) + np.sin(x), np.cos(y) + np.sin(y)) + 8.0
    fxy0 = -1.0*((((omega**2)*np.log(cxy)) - 2.0*np.multiply(np.cos(x) + np.sin(x), np.cos(y) + np.sin(y))))
    #V_0 = np.zeros((N1,N2))

    #Set up Verlet Method
    E_oldk = sfft.fft2(fxy0)
    E_curk = sfft.fft2(fxy0*np.cos(omega*dt))
    E_newk = E_newk = np.zeros((N1,N2))

    for tt in np.arange(2, N_t):
        #Verlet
        RHSk = spectralLaplacian(E_curk, ik1, ik2, cxy) + sfft.fft2(fxy0*np.cos(omega*tt*dt))
        E_newk =  2.0*E_curk + (dt**2)*RHSk - E_oldk
        E_oldk = E_curk
        E_curk = E_newk
    
    Exyt = np.real(sfft.ifft2(E_curk))
    return np.log(cxy)

def main():
    L = 2.0*PI
    N1 = N2 = 256 #Assume N1 = N2 in 2D FFT
    f1 = 1.0/(4.0*PI*np.sqrt(eps_0*u_0))
    Tf = 1.0/(PI*f1)
    omega = 100
    xc = np.linspace(0, L, num=N1)
    x, y = np.meshgrid(xc, xc, indexing='ij')
    fig, ax = plt.subplots(1,2)

    #Analytic solution
    cxy = np.multiply(np.cos(x) + np.sin(x), np.cos(y) + np.sin(y)) + 8.0
    E_true = np.log(cxy)*np.cos(omega*Tf)
    cs1 = ax[0].contourf(x, y, E_true)
    cbar1 = fig.colorbar(cs1)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title("Analytic Solution")
    
    #Numerical Solution
    En = waveEqSolve2D(N1, L, Tf)
    cs3 = ax[1].contourf(x, y, En)
    cbar3 = fig.colorbar(cs3)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title("Numerical Solution")

    plt.suptitle('Analytic vs. Numerical Solution of Electric Field E(x,y,Tf)')
    plt.show()

if __name__ == "__main__":
    main()