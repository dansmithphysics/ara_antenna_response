import math
import numpy as np
import matplotlib.pyplot as plt
import ROOT as root
import copy 
import scipy
from scipy.signal import savgol_filter

import sys
sys.path.append("./..")
import response_tools as rt

ff = 1.5e9 # Sampling speed of ARA
high_f = 7.5e8 
low_f =  1.3e8 
bPlot = True
freq_pad_len = 10 # len(fft) = 2**freq_pad_len
rc = np.sqrt(55.0*55.0 + 1.0 * 1.0) * (299792458.0 / 1.74) * 2.0 * np.pi / 4.0 # m

sim_gains, sim_freqs = rt.load_sim("../simulation/sim_graphs_all.root", "cormag_90")

cal_output = rt.load_cal_output("./cal_pulse_templates.root");
cal_output = cal_output[1:-1] # Make 512 elements, fftw made it weird 514
temp_cal_output = np.append(cal_output, np.zeros(2**freq_pad_len - len(cal_output)))
cal_output = temp_cal_output

# x axis values
t = np.array([i / ff for i in range(len(cal_output))]) * 1e9 # Units of ns
t -= t[-1] / 2.0
freqs = rt.freq_calc(len(cal_output), 1.0/ff) # Units of GHz

test_input = np.array(rt.load_test_input("/home/danielsmith/Summer2018/system_response/oct16_final_cleanup/testpulser_input_pulse.root", t))
test_output = np.array(rt.load_test_output("/home/danielsmith/Summer2018/system_response/average_of_test_pulses/system_response/exterpolate_master_pulse_template.root"))

# cal_input is such a sharp signal, going to idealize it as 1 value
temp_cal_input = np.zeros(2**freq_pad_len)
cal_input = temp_cal_input
cal_input[0] = -2.7 # To ensure correct power
temp_test_input = np.append(test_input, np.zeros(2**freq_pad_len - len(test_input)))
test_input = temp_test_input
temp_test_output = np.append(test_output, np.zeros(2**freq_pad_len - len(test_output)))
test_output = temp_test_output

# convert to volts at the digitizer
test_output *= (6.8 * 1e-3) 
cal_output  *= (6.8 * 1e-3) 

# Roll them to align with zero, done by eye
test_input = rt.align_with_zero(test_input)
test_output = rt.align_with_zero(test_output)
cal_output = rt.align_with_zero(cal_output)

#cal_output = np.roll(cal_output, 50)
#cal_output = [cal_output[i] if range(len(cal_output))[i] < 360 else 0.0 for i in range(len(cal_output))]

#plt.plot(range(len(cal_output)), cal_output)
#plt.plot(range(len(cal_input)), cal_input)
#plt.show()


# Get rid of stuff near causal boundaries
# Need to find new values for these guys
#cal_output  = rt.clean_limits(cal_output, 100.0, 19000.0, 25.0, 200.0)
#test_output = rt.clean_limits(test_output, 100.0, 19000.0, 25.0, 200.0)
#test_input  = rt.clean_limits(test_input, 100.0, 19000.0, 25.0, 200.0)

# Going to zero pad to ... 

cal_output_fft  = np.fft.rfft(cal_output)
cal_input_fft   = np.fft.rfft(cal_input)
test_output_fft = np.fft.rfft(test_output)
test_output_fft = np.append(test_output_fft, np.zeros(len(cal_output_fft) - len(test_output_fft)))
test_output     = np.fft.irfft(test_output_fft)
derivative_fft  = (1.0j * freqs) 
test_input_fft  = np.fft.rfft(test_input)
test_input_fft  = np.append(test_input_fft, np.zeros(len(test_output_fft) - len(test_input_fft)))

# subtract off att. at cal pulser
cal_input_fft *= np.power(10.0, -8.0/20.0) 

cal_output_fft *= np.power(10.0, 2.0 / 20.0) # correct for 2dB down from variable atten.

# Subtract off DC components
cal_output_fft[0]  = 0.0+0.0j
test_output_fft[0] = 0.0+0.0j
test_input_fft[0]  = 0.0+0.0j
derivative_fft[0]  = 0.0+0.0j

test_input = np.fft.irfft(test_input_fft)

# Calculate sys_fft
sys_fft = [test_output_fft[i] / test_input_fft[i] if test_input_fft[i] != 0.0 else 0.0 for i in range(len(test_input_fft))]
# add on system ampl
sys_fft = np.asarray(sys_fft)
sys_fft *= np.power(10.0, 87.0/20.0)

maybe_good_sys_fft = copy.deepcopy(sys_fft)

# Calculate h_fft
h_fft = []
for i in range(len(cal_output_fft)):
    if(sys_fft[i]*derivative_fft[i]*cal_input_fft[i] != 0.0):
        h_fft += [cal_output_fft[i] / (sys_fft[i] * derivative_fft[i] * cal_input_fft[i])]
    else:
        h_fft += [0.0]
h_fft = np.asarray(h_fft)
h_fft *= rc
h_fft = rt.fft_sqrt(h_fft, freqs)

h_fft = rt.smooth(h_fft, freqs, low_f, high_f)

# Zero out stuff
h_fft = [h_fft[i] if freqs[i] > 100e6 else 0.0 for i in range(len(freqs))]

# About to fit to a causal function
freqs_oi = rt.freq_calc(512, 1.0/1.5e9)

def disc_model(t, coeffs):
    temp = np.fft.fftshift(coeffs)
    tt = np.asarray([float(i) / 1.5e9 for i in range(len(temp))])
    tt -= tt[-1]/2.0
    #temp = [temp[i] if tt[i] > 0.0 else 0.0 for i in range(len(coeffs))]
    #temp = [temp[i] if tt[i] > 0.24e-8 else 0.0 for i in range(len(coeffs))]
    #temp = [temp[i] if tt[i] > 0.20e-8 else 0.0 for i in range(len(coeffs))]
    temp = [temp[i] if tt[i] > 0.20e-8 else 0.0 for i in range(len(coeffs))]
    #temp = [temp[i] if tt[i] > 0.0e-8 else 0.0 for i in range(len(coeffs))]
    return np.fft.fftshift(temp)

def disc_residuals(coeffs, y, t):
    model = disc_model(t, coeffs)
    res = np.fft.rfft(y) - np.fft.rfft(model)
    res /= freqs_oi#*freqs_oi # Weighted by frequency
    res = np.fft.irfft(res)
    res *= np.std(res)
    res *= np.std(model)
    return res

h_freq = np.fft.rfftfreq(512, 1.0 / 1.5e9)

# return to original length of h_fft
h = np.fft.irfft(h_fft)
h = np.copy(h[:512]) 
h_fft = np.fft.rfft(h)

x1_fft = h_fft;
x1 = np.fft.irfft(x1_fft)    
target = h
steps = range(len(target))

print "Starting Fit."
x, flag = scipy.optimize.leastsq(disc_residuals, x1, args=(target, steps))
print "Fit finished."

h = np.fft.fftshift(disc_model(steps, x))
t_h = np.array([i / 1.5e9 for i in range(len(np.fft.irfft(h_fft)))])
t_h -= t_h[-1] / 2.0
#h = np.array([h[i] if t_h[i] < 2.0e-8 and t_h[i] > 0.255e-8 else 0.0 for i in range(len(h))])
h = np.array([h[i] if t_h[i] < 3.0e-8 and t_h[i] > 0.2e-8 else 0.0 for i in range(len(h))]) #, good that I showed Abby
h = np.fft.fftshift(h)
h_fft = np.fft.rfft(h)

#h_fft = [h_fft[i] if h_freq[i] > 0.1e9 else 0.0 for i in range(len(h_fft))]

# Going to make sys_fft the Right length
#sys_graph = root.TGraph(len(sys_fft))
#for i in range(len(sys_fft)):
#    sys_graph.GetX()[i] = freqs[i]
#    sys_graph.GetY()[i] = np.absolute(sys_fft[i])
#sys_fft = np.array([sys_graph.Eval(h_freq[i]) for i in range(len(h_freq))]) #np.fft.rfft(np.fft.irfft(sys_fft)[:512])

#plt.plot(range(len(sys_fft)), 10.0 * np.log10(np.absolute(sys_fft)**2 * 2.0))
#plt.plot(range(len(maybe_good_sys_fft)), 10.0 * np.log10(np.absolute(maybe_good_sys_fft)**2 * 2.0))
#plt.show()
#plt.plot(range(len(np.fft.irfft(sys_fft))), np.fft.fftshift(np.fft.irfft(sys_fft)))
#plt.show()

h_fft = h_fft.astype("complex128")
sys_fft = sys_fft.astype("complex128")
#h_freq = h_freq.astype("float")

sys = np.fft.irfft(sys_fft)
h = np.fft.irfft(h_fft)
sys = rt.align_with_zero(sys)
h = rt.align_with_zero(h)

sys = sys[:len(h)]

sys_fft = np.fft.rfft(sys)
h_fft = np.fft.rfft(h)

np.save('ara_antenna_response', np.array(list(zip(freqs,h_fft))))
np.save('ara_system_response', np.array(list(zip(freqs,sys_fft))))

#final_list = []
#elect_list = []
#for i in range(len(h_fft)):
#    final_list += [[h_freq[i], h_fft[i]]]
#    elect_list += [[h_freq[i], sys_fft[i]]]
#final_list = np.asarray(final_list)
#elect_list = np.asarray(elect_list)
#np.save("ara_antenna_response", final_list)
#np.save("ara_system_response", elect_list)

# So... lets convolve them and see
convolo = h_fft * sys_fft[:len(h_fft)]
plt.plot(range(len(np.fft.irfft(convolo))), np.fft.fftshift(np.fft.irfft(convolo)))
plt.plot(range(len(np.fft.irfft(h_fft))), np.fft.fftshift(np.fft.irfft(h_fft)))
plt.show()

exit()

if(bPlot):
    '''
    plt.plot(freqs, np.unwrap(np.angle(h_fft)), label="h_fft")
    plt.plot(freqs, np.unwrap(np.angle(sys_fft)), label="sys_fft")
    plt.plot(freqs, np.unwrap(np.angle(test_input_fft)), label="test_input_fft")
    plt.plot(freqs, np.unwrap(np.angle(test_output_fft)), label="test_output_fft")
    plt.plot(freqs, np.unwrap(np.angle(cal_input_fft)), label="cal_input_fft")
    plt.plot(freqs, np.unwrap(np.angle(cal_output_fft)), label="cal_output_fft")
    plt.legend();
    plt.xlabel("Freq.")
    plt.xlim(0, 1e9)
    plt.ylim(0, -1400.0)
    plt.show()
    '''

    '''
    plt.plot(h_freq_trunc, 20.0 * np.log10( np.absolute(h_fft)), label="h_fft")
    plt.plot(freqs, 20.0 * np.log10( np.absolute(sys_fft)), label="sys_fft")
    plt.plot(freqs, 20.0 * np.log10( np.absolute(test_input_fft)), label="test_input_fft")
    plt.plot(freqs, 20.0 * np.log10( np.absolute(test_output_fft)), label="test_output_fft")
    plt.plot(freqs, 20.0 * np.log10( np.absolute(cal_input_fft)), label="cal_input_fft")
    plt.plot(freqs, 20.0 * np.log10( np.absolute(cal_output_fft)), label="cal_output_fft")
    plt.legend();
    plt.xlabel("Freq.")
    plt.xlim(0, 1e9)
    plt.ylim(-40.0, -10.0)
    plt.show()
    '''

    t_h = np.array([i / 1.5e9 for i in range(len(np.fft.irfft(h_fft)))])
    t_h -= t_h[-1] / 2.0

    plt.plot(t_h, np.fft.fftshift(np.fft.irfft(h_fft)), label="Data")
    plt.xlabel("Time [s]")
    plt.ylabel("Impulse Response [m]")
    plt.grid()
    plt.show()

    '''
    plt.title("Time-domain inputs to response")
    plt.plot(t, np.fft.fftshift(cal_output), label="cal_output")
    plt.plot(t, np.fft.fftshift(test_output), label="test_output")
    plt.plot(t, np.fft.fftshift(test_input), label="test_input")
    plt.plot(t_h, np.fft.fftshift(np.fft.irfft(h_fft)), label="Data")
    plt.xlabel("Time [ns]")
    plt.ylabel("Volts")
    plt.legend()
    plt.show()
    '''
    sim_height = np.power(10.0, np.array(sim_gains) / 10.0)
    sim_height /= 4.0 * np.pi
    sim_height /= (np.array(sim_freqs)*1e9 * 1.74 / 3e8)*( np.array(sim_freqs)*1e9 * 1.74 / 3e8)
    sim_height = np.sqrt(sim_height)

    plt.title("Antenna Effective Height")
    plt.plot(h_freq/1e9, np.absolute(h_fft) * np.sqrt(2), label="Data")
    plt.plot(sim_freqs, sim_height, label="xfdtd")
    plt.legend()
    plt.ylabel("Effective Height [m]")
    plt.xlabel("Freq. [GHz]")
    plt.xlim(0, 1.0)
    plt.grid()
    plt.show()    

    gain = copy.deepcopy(h_fft)
    gain *= h_freq * 1.74 / (3e8)
    gain *= np.conjugate(gain) * 2.0
    gain *= 4.0 * np.pi

    sys_gain = copy.deepcopy(sys_fft) #/ ff * (10e10/6.0)
    #sys_gain *= freqs[:len(sys_fft)] * 1.74 / (3e8)
    sys_gain *= np.conjugate(sys_gain) * 2.0
    #sys_gain *= 4.0 * np.pi

    plt.title("Antenna vs. Simulation Realized Gain")
    plt.plot(sim_freqs, sim_gains, label = "Simulation");
    plt.plot(h_freq/1e9, 10.0 * np.log10(np.absolute(gain)), label="Data Gain")
    plt.plot(freqs[:len(sys_gain)]/1e9, 10.0 * np.log10(np.absolute(sys_gain)), label="Sys. Gain")
    plt.xlim(0, 0.8)
    plt.ylim(-20, 80)
    plt.grid(which="both")
    
    plt.legend(loc="upper right")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Realized Gain [dB]")
    plt.show()

    plt.title("Antenna and system resposne.")
    plt.plot(h_freq, 10.0 * np.log10(np.absolute(h_fft)));
    plt.plot(freqs[:len(sys_fft)], 10.0 * np.log10(np.absolute(sys_fft)));
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Response [dB]")
    plt.ylim(-20.0, 80.0)
    plt.xlim(0, 1e9)
    plt.show();
