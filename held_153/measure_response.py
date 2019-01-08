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

ff = 1.5e11
high_f = 7.5e8 
low_f =  1.3e8 
bPlot = True
#rc = np.sqrt(55.0*55.0) * (299792458.0 / 1.74) * 2.0 * np.pi / 4.0 # m
#rc = np.sqrt(55.0*55.0 + 46.14*46.14) * (299792458.0 / 1.74) * 2.0 * np.pi / 4.0 # m
rc = np.sqrt(55.0*55.0 + 20.14*20.14) * (299792458.0 / 1.74) * 2.0 * np.pi / 4.0 # m

sim_gains, sim_freqs = rt.load_sim("../simulation/sim_graphs_all.root", "cormag_70")
cal_output = rt.load_cal_output("./cal_pulse_templates.root");

# x axis values
t = np.linspace(-327.680, 327.680, len(cal_output))
freqs = rt.freq_calc(len(cal_output), 1.0/ff)

cal_input = np.array(rt.load_cal_input("/home/danielsmith/Summer2018/system_response/oct16_final_cleanup/calpulser_input_spectrum.root", freqs))
test_input = np.array(rt.load_test_input("/home/danielsmith/Summer2018/system_response/oct16_final_cleanup/testpulser_input_pulse.root", t))
test_output = np.array(rt.load_test_output("/home/danielsmith/Summer2018/system_response/average_of_test_pulses/system_response/master_pulse_template.root"))

# convert to volts at the digitizer
test_output *= (6.8 * 1e-3) 
cal_output  *= (6.8 * 1e-3) 

# Roll them to align with zero, done by eye
test_input = rt.align_with_zero(test_input)
test_output = rt.align_with_zero(test_output)
cal_output = rt.align_with_zero(cal_output)

# Get rid of stuff near causal boundaries
cal_output  = rt.clean_limits(cal_output, 100.0, 19000.0, 25.0, 200.0)
test_output = rt.clean_limits(test_output, 100.0, 19000.0, 25.0, 200.0)
test_input  = rt.clean_limits(test_input, 100.0, 19000.0, 25.0, 200.0)

cal_output_fft  = np.fft.rfft(cal_output)
cal_input_fft   = np.fft.rfft(cal_input)
test_output_fft = np.fft.rfft(test_output)
test_output_fft = np.append(test_output_fft, np.zeros(len(cal_output_fft) - len(test_output_fft)))
test_output     = np.fft.irfft(test_output_fft)
derivative_fft  = (1.0j * freqs) 
test_input_fft  = np.fft.rfft(test_input)
test_input_fft  = np.append(test_input_fft, np.zeros(len(test_output_fft) - len(test_input_fft)))

# subtract off att. at cal pulser
cal_input_fft /= np.power(10.0, 8.0/20.0)

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
sys_fft *= np.power(10.0, 70.0/20.0) # not 70, due to variable attenuator

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

# Causal requirement
h_fft = rt.smooth(h_fft, freqs, low_f, high_f)

# Scale to match MC
h_fft /= np.power(10.0, 16.0/20.0)

# Causal requirement
h_fft = np.real(h_fft) + 1j * scipy.fftpack.hilbert(np.real(h_fft))

final_list = []
elect_list = []
for i in range(len(h_fft)):
    final_list += [[freqs[i], h_fft[i]]]
    elect_list += [[freqs[i], sys_fft[i]]]
final_list = np.asarray(final_list)
elect_list = np.asarray(elect_list)

np.save("ara_antenna_response", final_list)
np.save("ara_system_response", elect_list)

if(bPlot):

    #plt.plot(freqs, np.unwrap(np.angle(h_fft)), label="h_fft")
    #plt.plot(freqs, np.unwrap(np.angle(sys_fft)), label="sys_fft")
    #plt.plot(freqs, np.unwrap(np.angle(test_input_fft)), label="test_input_fft")
    #plt.plot(freqs, np.unwrap(np.angle(test_output_fft)), label="test_output_fft")
    #plt.plot(freqs, np.unwrap(np.angle(cal_input_fft)), label="cal_input_fft")
    #plt.plot(freqs, np.unwrap(np.angle(cal_output_fft)), label="cal_output_fft")
    #plt.legend();
    #plt.xlabel("Freq.")
    #plt.xlim(0, 1e9)
    #plt.ylim(0, -1400.0)
    #plt.show()

    #plt.plot(freqs, 20.0 * np.log10( np.absolute(h_fft)), label="h_fft")
    #plt.plot(freqs, 20.0 * np.log10( np.absolute(sys_fft)), label="sys_fft")
    #plt.plot(freqs, 20.0 * np.log10( np.absolute(test_input_fft)), label="test_input_fft")
    #plt.plot(freqs, 20.0 * np.log10( np.absolute(test_output_fft)), label="test_output_fft")
    #plt.plot(freqs, 20.0 * np.log10( np.absolute(cal_input_fft)), label="cal_input_fft")
    #plt.plot(freqs, 20.0 * np.log10( np.absolute(cal_output_fft)), label="cal_output_fft")
    #plt.legend();
    #plt.xlabel("Freq.")
    #plt.xlim(0, 1e9)
    #plt.ylim(-40.0, -10.0)
    #plt.show()

    plt.title("Time-domain inputs to response")
    plt.plot(t, np.fft.fftshift(cal_output), label="cal_output")
    plt.plot(t, np.fft.fftshift(test_output), label="test_output")
    plt.plot(t, np.fft.fftshift(test_input), label="test_input")
    plt.xlabel("Time [ns]")
    plt.ylabel("Volts")
    plt.legend()
    plt.show()

    gain = copy.deepcopy(h_fft) / ff * (10e10/6.0)
    gain *= freqs[:len(h_fft)] * 1.74 / (3e8)
    gain *= np.conjugate(gain) * 2.0
    gain *= 4.0 * np.pi

    sys_gain = copy.deepcopy(sys_fft) / ff * (10e10/6.0)
    sys_gain *= freqs[:len(sys_fft)] * 1.74 / (3e8)
    sys_gain *= np.conjugate(sys_gain) * 2.0
    sys_gain *= 4.0 * np.pi

    plt.title("Antenna vs. Simulation Realized Gain")
    plt.plot(sim_freqs, sim_gains, label = "Simulation");
    plt.plot(freqs[:len(gain)]/1e9, 10.0 * np.log10(np.absolute(gain)), label="Data Gain")
    plt.plot(freqs[:len(sys_gain)]/1e9, 10.0 * np.log10(np.absolute(sys_gain)), label="Sys. Gain")
    plt.xlim(0, 0.8)
    plt.ylim(-20, 15)
    plt.legend(loc="upper right")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Realized Gain [dB]")
    plt.show()

    plt.title("Antenna and system resposne.")
    plt.plot(freqs[:len(h_fft)], 10.0 * np.log10(np.absolute(h_fft)));
    plt.plot(freqs[:len(sys_fft)], 10.0 * np.log10(np.absolute(sys_fft)));
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Response [dB]")
    plt.ylim(-20.0, 80.0)
    plt.xlim(0, 1e9)
    plt.show();
