import numpy as np
import matplotlib.pyplot as plt

def MovingSurfGen3D(x, y, t, sigma, lM, lm, SpecSlope, beta, 
                    Usurf, d, GravityWaves):
    '''python port of Dr. Dolcetti's MATLAB code'''
    
    g = 9.81
    surface_tension = 72.75e-03 #N/m
    density = 998.2 #kg/m^3
    
    Nt = len(t)
    t = np.tile(np.array(t),[len(x),len(y),1])
    t = np.reshape(t,(Nt, len(x), len(y)))

    Dx = x[1] - x[0]
    Dy = y[1] - y[0]
    Dt = t[1] - t[0]
    
    ksx = 2*np.pi / Dx
    Dkx = (ksx / len(x))
    kx = np.arange(int(-ksx/2),int(ksx/2),Dkx) 
    
    ksy = 2*np.pi / Dy
    Dky = (ksy / len(y))
    ky = np.arange(int(-ksy/2),int(ksy/2),Dky)
    #ky = ky - ky[int(np.ceil(len(kx)/2)+1)]
    
    [Kx, Ky] = np.meshgrid(kx, ky)
    Kx = np.tile(Kx,[Nt,1,1])
    Ky = np.tile(Ky,[Nt,1,1])
    K = np.sqrt(Kx**2 + Ky**2)

    theta = np.arctan2(Ky,Kx)
    
    #Surface spectrum

    A = (K**(-SpecSlope/2))*np.exp(-beta*(np.cos(theta)+1)/2)
    A[np.where(np.abs(K) > 2*np.pi/lm)] = 0
    A[np.where(np.abs(K) < 2*np.pi/lM)] = 0

    
    Rand1 = np.random.randn(A.shape[1],A.shape[2]) + 1j*np.random.randn(A.shape[1],A.shape[2])
    Rand2 = np.random.randn(A.shape[1],A.shape[2]) + 1j*np.random.randn(A.shape[1],A.shape[2])
    Rand3 = np.random.randn(A.shape[1],A.shape[2]) + 1j*np.random.randn(A.shape[1],A.shape[2])


    Rand1 = np.tile(Rand1,[Nt,1,1])
    Rand2 = np.tile(Rand2,[Nt,1,1])
    Rand3 = np.tile(Rand3,[Nt,1,1])
    
    OmTurb = Usurf[0]*Kx + Usurf[1]*Ky
    OmGw = OmTurb + np.sqrt((g + surface_tension/density*K**2)*K*np.tanh(K*d))
    OmGw2 = OmTurb - np.sqrt((g + surface_tension/density*K**2)*K*np.tanh(K*d))
    
    spec1 = np.fft.ifftshift(Rand1 * A * np.exp(-1j * OmTurb * t))
    spec2 = np.fft.ifftshift(Rand2 * A * np.exp(-1j * OmGw * t))
    spec3 = np.fft.ifftshift(Rand3 * A * np.exp(-1j * OmGw2 * t))

    surf1 = np.fft.ifft(np.fft.ifft(spec2, axis = 1),axis = 2)
    surf2 = np.fft.ifft(np.fft.ifft(spec2, axis = 1),axis = 2)
    surf3 = np.fft.ifft(np.fft.ifft(spec3, axis = 1),axis = 2)

    zeta = np.real(surf1 + surf2 + surf3)

    zeta = zeta/np.mean(np.mean(np.std(zeta, axis=2),axis=1),axis=0)*sigma
    return zeta
    
