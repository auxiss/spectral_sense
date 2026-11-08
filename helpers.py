import numpy as np
import utils



def fft(samples, sample_rate, center_freq):   # numpy

    # Compute the FFT
    fft_result = np.fft.fft(samples)
    fft_freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)

    # Shift the center frequency component to the center
    fft_result = np.fft.fftshift(fft_result)
    fft_freqs = np.fft.fftshift(fft_freqs)
    fft_freqs += center_freq

    # Convert the FFT result to power (dB)
    epsilon = 1e-12
    power_spectrum = 20 * np.log10(np.abs(fft_result) + epsilon)
    #print(power_spectrum)

    return fft_freqs , power_spectrum




def find_peaks_with_neighbors(data, neighbors=1):

    peaks = []
    length = len(data)
    
    for i in range(neighbors, length - neighbors):
        # Compare the current point to its neighbors
        is_peak = True
        for j in range(1, neighbors + 1):
            if data[i] <= data[i - j] or data[i] <= data[i + j]:
                is_peak = False
                break
        if is_peak:
            peaks.append(i)
    
    return peaks



def find_approx_index(table, value, tolerance):
    low, high = 0, len(table) - 1
    while low <= high:
        mid = (low + high) // 2
        diff = abs(table[mid] - value)
        if diff <= tolerance:
            return mid
        elif table[mid] < value:
            low = mid + 1
        else:
            high = mid - 1

            
    # Return the closest index instead of raising an error
    if low >= len(table):
        return len(table) - 1
    if high < 0:
        return 0
    raise LookupError



def find_intex_of_max_in_range(data, table, start, end):
    start = find_approx_index(table, start, 0)
    end = find_approx_index(table, end, 0)
    max_value = float('-inf')
    max_index = -1

    for i in range(int(start), int(end)):
        if data[i] > max_value:
            max_value = data[i]
            max_index = i

    return max_index, max_value




def remove_artifacts(fft_freqs , power_spectrum, dc_spike_cut=5e3, outer_cut=200e3):
    fft_freqs = np.asarray(fft_freqs)
    power_spectrum = np.asarray(power_spectrum)

    if fft_freqs.shape != power_spectrum.shape:
        raise ValueError("fft_freqs and power_spectrum must have the same shape")

    center_freq = (fft_freqs[0] + fft_freqs[-1]) / 2
    half_bandwidth = (fft_freqs[-1] - fft_freqs[0]) / 2
    valid_band = np.abs(fft_freqs - center_freq) <= half_bandwidth - outer_cut
    outside_dc_spike = np.abs(fft_freqs - center_freq) >= dc_spike_cut / 2
    valid = valid_band & outside_dc_spike

    return fft_freqs[valid], power_spectrum[valid]



def avrg_power(power_frame):
            sum = 0
            for power in power_frame:
                sum += power
            sum = sum / len(power_frame)
            return sum

def max_power(power_frame):
    max_p = 0
    for power in power_frame:
        if max_p < power: max_p = power
    return max_p


def finde_beacons(power_frame, frequencies, threshold):
    #print("Finding beacons...")

    beacons = []
    threshold_values = np.asarray(threshold)
    if threshold_values.ndim > 0 and threshold_values.shape != np.shape(power_frame):
        raise ValueError("threshold must be a scalar or match power_frame")

    reciving = False
    for i in range(len(power_frame)):
        threshold_value = threshold_values if threshold_values.ndim == 0 else threshold_values[i]
        if power_frame[i] > threshold_value and not reciving:
            reciving = True
            start_index = i
        elif power_frame[i] < threshold_value and reciving:
            reciving = False
            end_index = i
            beacon_freq = frequencies[(start_index + end_index) // 2]
            bandwidth = frequencies[end_index] - frequencies[start_index]

            beacon = {
                "frequency": beacon_freq,
                "bandwidth": bandwidth,
                "start_index": start_index,
                "end_index": end_index
            }


            beacons.append(beacon)
            #print(f"Beacon found at {utils.freq_to_text(beacon['frequency'])}Hz, BW: {utils.freq_to_text(beacon['bandwidth'])}.")



    return beacons




from scipy import signal

def band_pass_filter(samples, sample_rate, start_freq, end_freq,
                     state=None, num_taps=129):
    """Streaming band-pass for complex IQ samples.
    Returns (filtered_samples, state). Pass state back in on the next buffer."""
    samples = np.asarray(samples, dtype=np.complex64)
    center = (start_freq + end_freq) / 2
    bandwidth = end_freq - start_freq

    if state is None:
        state = {
            "taps": signal.firwin(num_taps, bandwidth / 2, fs=sample_rate),
            "zi": np.zeros(num_taps - 1, dtype=np.complex64),
            "phase": 0.0,
        }

    # Phase ramp that shifts the band center down to 0 Hz, continuous across buffers
    step = -2 * np.pi * center / sample_rate
    ph = state["phase"] + step * np.arange(len(samples))

    shifted = samples * np.exp(1j * ph)
    filtered, state["zi"] = signal.lfilter(state["taps"], 1.0, shifted, zi=state["zi"])
    result = filtered * np.exp(-1j * ph)  # shift back to the original band

    state["phase"] = (state["phase"] + step * len(samples)) % (2 * np.pi)
    return result, state

    


def avr_power_of_cmplex(samples):
    """Calculate the average power of complex IQ samples output in db"""
    power = np.mean(np.abs(samples) ** 2)
    power_db = 10 * np.log10(power + 1e-12)  # Add a small value to avoid log(0)
    return power_db





def decimate(samples, factor):
    """Decimate the samples by the given factor."""
    return signal.decimate(samples, factor, ftype='fir', zero_phase=True)




def local_mean_curve(power, n_neighbors):
    """
    Mean of the n_neighbors bins on each side of every point (the point itself
    is excluded, so a strong peak doesn't raise its own threshold).
    Edges are padded by repeating the first/last value.
    """
    power = np.asarray(power, dtype=float)
    kernel = np.ones(2 * n_neighbors + 1)
    kernel[n_neighbors] = 0            # exclude the centre bin
    kernel /= kernel.sum()
    padded = np.pad(power, n_neighbors, mode="edge")
    return np.convolve(padded, kernel, mode="valid")   # same length as power