import sys
import numpy as np
import matplotlib.pyplot as plt

# Antenna response
file_ant = np.load(sys.argv[1])

# Electronics response
file_elect = np.load(sys.argv[2])

freqs, h_fft = np.hsplit(file_ant, 2)
freqs, sys_fft = np.hsplit(file_elect, 2)
h_fft = np.ravel(h_fft)
sys_fft = np.ravel(sys_fft)

plt.plot(freqs, 20.0 * np.log10(np.absolute(h_fft)))
plt.plot(freqs, 20.0 * np.log10(np.absolute(sys_fft)))
plt.xlabel("Freq. [Hz]")
plt.ylabel("dB")
plt.xlim(0, 1000e6)
plt.show()

h = np.fft.irfft(h_fft)
sys = np.fft.irfft(sys_fft)

plt.plot(range(len(h)), np.fft.fftshift(h))
plt.xlabel("Time [NOT NS]")
plt.ylabel("abu")
plt.show()

