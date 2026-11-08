import numpy as np
from scipy.interpolate import interp1d
import helpers as mf


class full_speqtrum:
    def __init__(self, common_freqs):

        self.full_power = []
        self.common_freqs = common_freqs

        self.rezolution = common_freqs[1] - common_freqs[0]
        self.current_frequencies = []
        self.current_power = []  
        

        for i in range(len(self.common_freqs)):
            self.full_power.append(0)

        #print(f'common freqs len: {len(self.common_freqs)}')
        #print(f'full power len: {len(self.full_power)}')
    
    def add(self, power_spectrum, frequencies):
        self.current_frequencies = frequencies
        self.current_power = power_spectrum
        self.center_freq = frequencies[len(frequencies) // 2]

        #print(f' first freq: {frequencies[1]}')
        #print(f' last freq: {frequencies[-1]}')
        
        # linear nearest zero slinear Quadratic cubic
        predict = interp1d(frequencies, power_spectrum, kind='linear', bounds_error=True)


        first_step = mf.find_approx_index(self.common_freqs, frequencies[1], self.rezolution) - 1
        last_step = mf.find_approx_index(self.common_freqs, frequencies[-1], self.rezolution) + 1

        #print(F'step range: {last_step - first_step}')
        


        # steach frame to full scan
        step = first_step
        while step <= last_step:
            try:
                if self.full_power[step] == 0:
                    self.full_power[step] = predict(self.common_freqs[step])

                else:
                    self.full_power[step] = (self.full_power[step] + predict(self.common_freqs[step])) / 2

            except:
                #print(self.full_power[step])
                pass
            
            step += 1

        


    def get_full_spectrum_power(self):
        return self.full_power

    def get_full_spectrum(self):
        return self.common_freqs

   
    