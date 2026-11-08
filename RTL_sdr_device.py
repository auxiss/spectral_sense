from rtlsdr import RtlSdr
import time






class rtl_sdr_device :
    def __init__(self):

        #init sdr
        print("init..")

        self.sdr = RtlSdr()
        self.sdr.sample_rate = 2.4e6
        self.sdr.center_freq = 100e6
        self.sdr.gain = 'auto'

        print(self.sdr)




    def set_sample_rate(self, sr):
       
        if not isinstance(sr, (int, float)):
            raise TypeError("Sample rate must be a number")
        
        if not (225001 <= sr <= 300000 or 900001 <= sr <= 3200000):
            raise ValueError("Sample rate must be from 225001Hz to 300000Hz")
        
        self.sdr.sample_rate = sr




    def set_center_freq(self, cf):
        
        if not isinstance(cf, (int, float)):
            raise TypeError("Center frequency must be a number")
        
        self.sdr.center_freq = cf




    def set_gain(self, gain):
            
        if not (type(gain) == int) and not (type(gain) == float) and not (gain == 'auto'):
            raise TypeError("Gain must be 'auto' or a number")

        self.sdr.gain = gain

    def set_gain_percent(self, gain_percent):
        if not isinstance(gain_percent, (int, float)):
            raise TypeError("Gain percent must be a number")
        
        if not (0 <= gain_percent <= 100):
            raise ValueError("Gain percent must be between 0 and 100")
        
        valid_gains = self.sdr.valid_gains_db
        gain_index = int((gain_percent / 100) * (len(valid_gains) - 1))
        self.sdr.gain = valid_gains[gain_index]

    def set_ppm_correction(self, ppm):
        self.sdr.freq_correction = ppm


    def close(self):
        self.sdr.close()




    def read_samples(self, num_samples=1024):
        return self.sdr.read_samples(num_samples)       




    def info(self, full=False):

        if full:

            info = {
                "sample_rate": self.sdr.sample_rate,
                "center_freq": self.sdr.center_freq,
                "gain": self.sdr.gain,
                "freq_correction": self.sdr.freq_correction,
                "bandwidth": self.sdr.bandwidth,
                "valid_gains_db": self.sdr.valid_gains_db,
                "gain_values": self.sdr.gain_values,
                "tuner_type": self.sdr.get_tuner_type(),
            }

        else:
            info = {
                "sample_rate": self.sdr.sample_rate,
                "center_freq": self.sdr.center_freq,
                "gain": self.sdr.gain,
                "freq_correction": self.sdr.freq_correction,
                "bandwidth": self.sdr.bandwidth,
            }

        return info




if __name__ == "__main__":

    SD = rtl_sdr_device()


    print(SD.info(full=False))


    num_samples = 1024
    start = time.perf_counter()
    samples = SD.read_samples(num_samples)
    end = time.perf_counter()
    print(f"sampling {num_samples} samples time: {end - start:.6f} seconds")


    SD.set_gain_in_persent(100)

    print(SD.info())

    SD.close()


