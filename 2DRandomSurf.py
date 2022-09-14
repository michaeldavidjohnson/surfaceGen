import numpy as np
def random_surface_fully_vec(beta, x, t, velocity, depth, lM, lm):
  g = 9.81
  surface_tension = 72.75e-03
  density = 998.2
  aw = 1e-3

  Dx = x[1] - x[0]
  Dt = t[1] - t[0]

  ksx = 2*np.pi / Dx
  omega = 2*np.pi / Dt

  omega = 2*np.pi / Dt

  Dkx = ksx / len(x)
  Domega = omega / len(t)

  Nt = np.arange(int(-omega/2), int(omega/2), Domega)

  Kx = np.arange(int(-ksx/2), int(ksx/2), Dkx)
  K_y = Kx

  
  Kx = np.tile(Kx,[int(len(Nt))+1,1])


  t = np.tile(np.array(t), [Kx.shape[1],1])
  t = np.swapaxes(t, 0, 1)

  A = (Kx**(-beta/2))

  A[np.where(Kx > 2*np.pi/lm)] = 0
  A[np.where(Kx < 2*np.pi/lM)] = 0

  
  turb = Kx*velocity
  gw1 = turb + np.sqrt((g + surface_tension/density*Kx**2)*Kx*np.tanh(Kx*depth))
  gw2 = turb - np.sqrt((g + surface_tension/density*Kx**2)*Kx*np.tanh(Kx*depth))

  rand1 = np.tile(np.random.randn(A.shape[1]),[A.shape[0],1])
  rand2 = np.tile(np.random.randn(A.shape[1]),[A.shape[0],1])
  rand3 = np.tile(np.random.randn(A.shape[1]),[A.shape[0],1])

  spec1 = np.fft.ifftshift(rand1 * A * np.exp(-1j * turb * t))
 
  spec2 = np.fft.ifftshift(rand2 * A * np.exp(-1j * gw1 * t))

  spec3 = np.fft.ifftshift(rand3 * A * np.exp(-1j * gw2 * t))

  hmm = np.fft.ifft(spec1 + spec2 + spec3,axis=1)
  surf1 = np.fft.ifft(spec1, axis = 1)
  surf2 = np.fft.ifft(spec2, axis = 1)
  surf3 = np.fft.ifft(spec3, axis = 1)
  
  total = (surf1 + surf2 + surf3)

  total = np.real(total)/np.std(total)*aw
  return total
