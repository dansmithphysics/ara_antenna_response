import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
import ROOT as root
import copy 
import scipy
from scipy import signal
from scipy.signal import savgol_filter

def freq_calc(n, dt):
    freq = []
    if n % 2 == 0:
        freq = np.linspace(0, 1 / (2. * dt), (n / 2) + 1)
    else:
        freq = np.linspace(0, (n - 1.) / (2. * dt * n), (n + 1) / 2)
    return freq

def fft_sqrt(fft, freqs):
    ''' Complex sqrt requires unwrapping phase. '''

    gain = np.absolute(fft)
    gain = np.sqrt(gain)    

    phase = np.angle(fft)
    phase = np.unwrap(phase)
    phase /= 2.0

    fft = np.cos(phase) * gain + np.sin(phase) * gain *1j
    return fft

def butterworth(fft, n, freqs, fc, low):
    ''' 
    Butterworth Filter   
    H(s) = G0 / prod(s - sk) / wc
    for sk = wc exp(j(2k+n-1) pi / (2n)
    low determines if it acting as a low pass or high pass filter
    '''
    w  = 2.0 * np.pi * freqs
    wc = 2.0 * np.pi * fc

    gain = np.absolute(fft)
    phase = np.angle(fft)

    if(low):
        gain *= np.power(np.absolute(w / wc), n) / np.sqrt(1.0 + np.power(w / wc, 2 * n))
    else:
        gain *= 1.0 / np.sqrt(1.0 + np.power(w / wc, 2 * n))
    return gain * (np.cos(phase) + 1j * np.sin(phase))
    

def smooth(fft, freqs, low_limit, high_limit):
    ''' Function to clean up areas of very low snr ''' 

    fft_og = copy.deepcopy(fft)

    # Cheat a bit here, looks like noise is causing issues
    first_good = 0.0
    for i in range(len(fft)):
        if(freqs[i] >= 1.40e8):
            first_good = fft[i]
            break
    for i in range(len(fft)):
        fft[i] = first_good
        if(freqs[i] >= 1.40e8):
            break

    last_good = 0.0
    for i in range(len(fft)):
        if(freqs[i] >= 7.40e8):
            last_good = fft[i]
            break
    for i in range(len(fft)):
        if(freqs[i] >= 7.40e8):
            fft[i] = last_good

    # First, deal with discont. at boundaries with butterworth's help
    fft = butterworth(fft, 10.0, freqs, low_limit, True)
    fft = butterworth(fft, 30.0, freqs, high_limit, False)

    abso = np.absolute(fft)
    phaso = np.unwrap(np.angle(fft))

    # Next, need to linearly extrapolate over the 450 MHz notch filter 
    #  because we can't read zero
    # Extrapolate over a 15 Mhz range around notch
    before, after = 0.0, 0.0
    before_phase, after_phase = 0.0, 0.0

    last = 0.0
    checked = False
    for i in range(len(abso)):
        if((freqs[i] - 4.5e8) <= -0.15e8):
            before = abso[i]
            before_phase = phaso[i]
        if((freqs[i] - 4.5e8) >= +0.15e8 and not(checked)):
            after = abso[i]
            after_phase = phaso[i]
            checked = True
        if(freqs[i] >= 7.5e8):
            last = abso[i]
            break

    abso = [abso[i] if (np.fabs(freqs[i] - 4.5e8) > 0.15e8) else 
            (after - before)/0.3e8 * (freqs[i] - 4.5e8) + (before+after)/2.0 for i in range(len(abso))]
    phaso= [phaso[i] if(np.fabs(freqs[i] - 4.5e8) > 0.15e8) else 
            (after_phase - before_phase)/0.3e8 * (freqs[i] - 4.5e8) + (before_phase+after_phase)/2.0 for i in range(len(phaso))]

    fft = abso*(np.cos(phaso) + 1j * np.sin(phaso))
    return fft

def smooth_freq(fft):
    abso = np.absolute(fft)
    phaso = (np.unwrap(np.angle(fft)))
    phaso = savgol_filter(phao, 111, 1)    
    fft = abso*(np.cos(phaso) + 1j * np.sin(phaso))
    return fft

def freq_cut(fft, freqs, low_f, high_f):
    return np.asarray([fft[i] if (freqs[i] > low_f and freqs[i] < high_f) else 0.0 for i in range(len(fft))])

def find_optimal(shiz, local):
    shiz_og = copy.deepcopy(shiz)
    valos = []
    for i in range(len(shiz)):
        valos += [np.average(np.array(range(len(shiz))) * shiz / np.sum(shiz))]
        shiz = np.roll(shiz, 1);
        if(len(valos) > 10):
            if(valos[-1] > valos[-10] and valos[-1] > valos[-2]):
                break;
    cur_closest = np.absolute(valos[0] - local);
    cur_index = 0;
    for i in range(len(valos)):
        if(cur_closest > np.absolute(valos[0] - local)):
           cur_index = i
           cur_cloest = np.absolute(valos[0] - local);
    return np.roll(shiz_og, i)

def load_sim(path, plot):
    f_sim = root.TFile(path)
    sim_gain_graph = f_sim.Get(plot)
    sim_freqs = [sim_gain_graph.GetX()[i] for i in range(sim_gain_graph.GetN())]
    sim_gains = [sim_gain_graph.GetY()[i] for i in range(sim_gain_graph.GetN())]
    return sim_gains, sim_freqs

def load_cal_output(path):
    f_cal_output = root.TFile(path)
    cal_output_graph = f_cal_output.Get("template_0")
    cal_output = np.asarray([cal_output_graph.GetY()[i] for i in range(cal_output_graph.GetN())])
    return cal_output

def load_cal_input(path, freqs):
    f_cal_input = root.TFile(path)
    cal_input_graph = f_cal_input.Get("Graph")
    cal_input_fft = [np.power(10.0, cal_input_graph.Eval(freqs[i]) / 10.0) for i in range(len(freqs))]
    return np.fft.irfft(cal_input_fft)

def load_test_input(path, t):
    f_test_input = root.TFile(path);
    test_input_graph = f_test_input.Get("wave")
    test_input = [test_input_graph.Eval(iT/1e9) if (iT > -5.0 and iT < 5.0) else 0.0 for iT in t]
    test_input = np.fft.fftshift(test_input)
    return test_input

def load_test_output(path):
    f_test_output = root.TFile(path);
    test_output_graph = f_test_output.Get("template_0")
    test_output = [test_output_graph.GetY()[i] for i in range(test_output_graph.GetN())]
    return test_output

def clean_limits(signal, lower, upper, tau_lower, tau_upper):
    for i in range(len(signal)):
        # zero out stuff close to late times
        if(i > upper):
            signal[i]  *= math.exp(-(i - upper) / tau_upper)        
        # zero out stuff before t=0
        if(i >= len(signal) // 2.0):
            signal[i]  = 0.0
        # Get rid of artifacts very close to t=0
        if(i >= 0 and i < lower):
            signal[i]  *= (1.0 - math.exp(-i/tau_lower))
    return signal;
    
def align_with_zero(signal): 
    ''' Finds where 2% of power is, and roughly aligns it with zero ''' 

    # Get rid of baseline as much as I can
    #signal -= np.average(signal) 

    running_total_power = 0.0
    power = np.zeros(len(signal))
    for i in range(len(signal)):
        running_total_power += signal[i]*signal[i]
        power[i] = running_total_power
    power /= running_total_power

    start_index = 0
    for i in range(len(power)):
        if(power[i] > 0.1):
            start_index = i
            break
    signal = np.roll(signal, -1 * start_index + int(len(signal) * 0.01))
    return signal
