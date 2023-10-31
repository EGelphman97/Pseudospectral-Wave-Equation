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

def spectralLaplacian7(Yk, ik1, ik2, c2):
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

def spectralLaplacian6(Yk, ik1, ik2, c2):
    """
    Function to calculate the Laplacian u = Del*((c^2(x1,x2))Del(y(x1,x2))) in the Fourier domain
    where Del is the two-dimensional gradient operator using Algorithm 6 from the paper "Notes on 
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
    N1 = Yk.shape[0]
    N2 = Yk.shape[1]
    Yk_nrow = Yk[N1//2,:]
    Yk_ncol = Yk[:,N2//2]
    ik1Yk = np.multiply(ik1, Yk)
    ik2Yk = np.multiply(ik2, Yk)
    gn1 = np.real(sfft.ifft2(ik1Yk))
    gn2 = np.real(sfft.ifft2(ik2Yk))
    #Step 2 - Calculate c^2(x,y)*Del(Yk) in time domain using inverse 2D FFT
    vn1 = np.multiply(c2, gn1)
    vn2 = np.multiply(c2, gn2)
    #print("Max of gn1: " + str(np.max(vn1)) + "   Max. of gn2: " + str(np.max(vn2)))
    #Step 3 - Calculate Auxiliary arrays from saved Nyquist data
    cmean_n2 = c2.mean(0)
    cn1_mean = c2.mean(1).T
    yhatrow = np.real(sfft.ifft(Yk_nrow))
    yhatcol = np.real(sfft.ifft(Yk_ncol))
    an1 = np.multiply(cn1_mean, yhatcol)
    an2 = np.multiply(cmean_n2, yhatrow)
    Ak1 = sfft.fft(an1)
    Ak2 = sfft.fft(an2)
    #Step 4 - Calculate Del dot Del(Yk) in frequency domain to get Spectral Laplacian
    #using auxiliary arrays
    Vk1 = sfft.fft2(vn1)
    Vk2 = sfft.fft2(vn2)
    Uk = np.multiply(ik1, Vk1) + np.multiply(ik2, Vk2) 
    Uk[(N1//2),:] += -1.0*((N1/PI)**2)*Ak2
    Uk[:,(N2//2)] += -1.0*((N2/PI)**2)*Ak1
    return Uk

def calcRHSk(Ek, ik1, ik2, cxy, Fk0, omega, t):
    """
    Function to calculate the RHS of the wave equation with source term and variable
    wave speed in the discrete frequency domain (k-domain)

    Parameters:
        Ek: Fourier transform of E(x,y,t)
        ik1, ik2: Matrices of discrete frequencies used in fft2
        cxy: Position-dependent wave speed
        Fk0: Fourier transform of f(x,y,0)
        omega: Frequency of source term
        t: Time step
    """
    RHSk = spectralLaplacian6(Ek, ik1, ik2, cxy) + Fk0*np.cos(omega*t)
    return RHSk

def waveEqSolve2D(N, L, Tf, omega):
    """
    Function to solve the two-dimensional wave equation for a scalar version of
    the electric field E(x,y,t) the two-dimensional using the FFT based Pseudo-
    Spectral method with Verlet time stepping

    Parameters:
        N : Square root of 2D FFT Size
        L: Length of One Side of Rectangle
        Tf: Final time value of numerical solution, in seconds
        omega: Oscillation frequency of source term       

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
    xc = h*np.arange(N1)
    x, y = np.meshgrid(xc, xc, indexing='ij')

    #Calculate the position-dependent wave velocity c(x,y) and source term f(x,y,t)
    cxy = np.multiply(np.cos(x, dtype="double") + np.sin(x, dtype="double"), np.cos(y,dtype="double") + np.sin(y,dtype="double")) + 8.0
    dt = 50.0*np.sqrt((u_0*eps_0*np.min(cxy))/(2.0*(np.max(k1)**2)), dtype="double") #dt_max/5
    N_t = int(Tf/dt + 0.5)#Number of time steps
    print("Number of time steps:" + str(N_t))
    fxy0 = -1.0*((omega**2)*np.log(cxy, dtype="double") - 2.0**np.multiply(np.cos(x,dtype="double") + np.sin(x,dtype="double"), np.cos(y,dtype="double") + np.sin(y,dtype="double")))
    Fk0 = sfft.fft2(fxy0)

    #Set up Verlet Method
    Ak = sfft.fft2(np.log(cxy, dtype="double"))
    Vk = sfft.fft2(np.zeros((N1,N2), dtype="double"))
    Aknew = Vknew = np.zeros((N1,N2), dtype="double")

    #Set up RK3 Method
    E1k = Ak
    E2k = E1newk = E2newk = np.zeros((N1,N2), dtype="double")

    for tt in np.arange(1, N_t):
        #Verlet
        RHSk = calcRHSk(Ak, ik1, ik2, cxy, Fk0, omega, tt*dt)
        Vknew = Vk + dt*RHSk
        Vk = Vknew
        Aknew = Ak + dt*Vk
        Ak =  Aknew

        #RK3
        K11 = E2k
        K12 = calcRHSk(E1k, ik1, ik2, cxy, Fk0, omega, tt*dt)
        K21 = E2k + 0.5*dt*K12
        K22 = calcRHSk(E1k + 0.5*dt*K11, ik1, ik2, cxy, Fk0, omega, tt*dt)
        K31 = E2k - dt*K12 + 2.0*dt*K22
        K32 = calcRHSk(E1k - dt*K11 + 2.0*dt*K21, ik1, ik2, cxy, Fk0, omega, tt*dt)
        E1newk = E1k + (dt/6.0)*(K11 + 4.0*K21 + K31)
        E2newk = E2k + (dt/6.0)*(K12 + 4.0*K22 + K32)
        #Increment time step
        E1k = E1newk
        E2k = E2newk

        if tt % 250 == 0:
            print("Time Step " + str(tt))
    
    ExytRK3 = np.real(sfft.ifft2(E1k))
    ExytV = np.real(sfft.ifft2(Ak))
    return ExytRK3, ExytV

def main():
    L = 2.0*PI
    N1 = N2 = 512 #Assume N1 = N2 in 2D FFT
    omega = 1e6#rad/s
    Tf = (2.0*PI/omega)  
    h = L/N1
    xc = h*np.arange(0,N1)
    x, y = np.meshgrid(xc, xc, indexing='ij')
    fig, ax = plt.subplots(1,3)

    #Analytic solution
    cxy = np.multiply(np.cos(x) + np.sin(x), np.cos(y) + np.sin(y)) + 8.0
    E_true = np.log(cxy)*np.cos(omega*Tf)
    cs1 = ax[0].contourf(x, y, E_true)
    cbar1 = fig.colorbar(cs1)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title("Analytic Solution")
    
    #Numerical Solution
    EnRK3, EnV = waveEqSolve2D(N1, L, Tf, omega)
    k_neg = np.arange(-N1//2 + 1,0,1)
    k_pos = np.concatenate((np.arange(N1//2), np.array([0])))
    k1 =  np.concatenate((k_pos, k_neg))
    ik = 1j*k1
    ik1, ik2 = np.meshgrid(ik, ik, indexing='ij')#Each k needs to be a 2d array/matrix/meshgrid
    cs2 = ax[1].contourf(x, y, EnRK3)
    cbar2 = fig.colorbar(cs2)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title("Numerical Solution - RK3")
    cs3 = ax[2].contourf(x, y, EnV)
    cbar3 = fig.colorbar(cs3)
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')
    ax[2].set_title("Numerical - Verlet")

    plt.suptitle('Numerical vs. Analytic Solution of Wave Equation With Source')
    plt.show()

if __name__ == "__main__":
    main()