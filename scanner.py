import RTL_sdr_device
import numpy as np
import helpers as mf
import threading
import swip
import utils
import time


class scanner:
    def __init__(self,sample_rate, 
                 min_freq, max_freq, 
                 smothnes=50, 
                 gain=0, 
                 threshold=7, 
                 beacon_bandwidth=10e3, 
                 outer_cut = 400e3, 
                 dc_spike_cut = 5e3,
                 dynamic_thershold = False,
                 neighbors = 100e6       # bins on EACH side used for the local mean
                 ):
        
        
        self.running = True



        self.max_hold_power = None

        self.full_spectrum_power = None
        self.full_spectrum = None

        self.beacons = []


        #sdr paramiters
        self.center_freq = min_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.sampleSize = 1024*4
        self.SD = RTL_sdr_device.rtl_sdr_device()
        self.SD.set_sample_rate(self.sample_rate)
        self.SD.set_center_freq(self.center_freq)
        self.SD.set_gain_percent(self.gain)
        print(f"sdr divece initialized")


        #config prams
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.beacon_bandwidth = beacon_bandwidth  # Hz
        self.starting_threshold = threshold       # user offset (dB)
        self.threshold = self.starting_threshold  # single number (mean of the curve)
        self.threshold_curve = None               # per-bin curve, set after the first sweep
        self.neighbors = self.sampleSize
        self.smothnes = smothnes #avring
        self.dynamic_thershold = dynamic_thershold

        

        self.outer_cut = outer_cut
        self.dc_spike_cut = dc_spike_cut
        self.step_size = (self.sample_rate / 2 - (self.dc_spike_cut + self.outer_cut))
        self.total_steps= self.max_freq / self.step_size

        print(f"\nstep size:\t\t {utils.freq_to_text(self.step_size)}")

        self.freq_resolution = self.sample_rate / self.sampleSize
        print(f"frequency resolution: {utils.freq_to_text(self.freq_resolution)}")



        self.common_freqs = np.arange(min_freq, max_freq, self.freq_resolution)
        self.swip = swip.full_speqtrum(common_freqs=self.common_freqs)



        

    def scan(self,  center_frequency_callback=None):
        for center_freq in np.arange(self.min_freq, self.max_freq, self.step_size):
            if not self.running:
                break

            if center_frequency_callback is not None:
                center_frequency_callback(center_freq)

            self.center_freq = float(center_freq)
            self.SD.set_center_freq(self.center_freq)
            #print(self.SD.info(), end='')

            self.percentage = int(((self.center_freq - self.min_freq) / (self.max_freq - self.min_freq)) * 100)
            #print(f" - {self.percentage}%")

            # arvrage samples
            power_sum = None
            for i in range(self.smothnes):

                iq = self.SD.read_samples(self.sampleSize)
                fft_freqs, power_spectrum = mf.fft(
                    samples=iq,
                    sample_rate=self.sample_rate,
                    center_freq=self.center_freq,
                )

                if power_sum is not None:
                    power_sum += power_spectrum
                else:
                    power_sum = power_spectrum


            if power_sum is not None:
                avrage_power = power_sum/self.smothnes
            else:
                print("wtff!!!!")
                exit(1)

            
                    

            filtered_freqs, filtered_power = mf.remove_artifacts(
                fft_freqs,
                avrage_power,
                dc_spike_cut=self.dc_spike_cut,
                outer_cut=self.outer_cut,
            )


            
            #self.finde_beacons(
            #    filtered_power, 
            #    filtered_freqs,
            #)

            self.swip.add(
                    power_spectrum=filtered_power,
                    frequencies=filtered_freqs
            )



        if self.running:
            #print("lap complited")
            self.full_spectrum_power = self.swip.get_full_spectrum_power().copy()
            self.full_spectrum = self.swip.get_full_spectrum()
            self._update_max_hold()
            self._update_threshold()

        

    def _update_max_hold(self):

        if self.max_hold_power is None:
            self.max_hold_power = self.full_spectrum_power.copy()
        else:
            self.max_hold_power = np.maximum(
                self.max_hold_power,
                self.full_spectrum_power,
            )

        


    def close(self):
        print("swiper shuting down")
        self.running = False
        self.SD.close()




    def get_frame(self):
        frame= {
            "power": self.full_spectrum_power,
            "freqs": self.full_spectrum,
            "max"  : self.max_hold_power,
            "threshold": self.threshold_curve,
        }
        return frame

    def get_current_frame(self):
        frame = {
            "power": self.swip.get_full_spectrum_power(),
            "freqs": self.swip.get_full_spectrum()
        }
        return frame

    def _update_threshold(self):
        """
        Builds the localized threshold curve:

            curve[i] = mean of the neighbouring max-hold bins + user offset

        self.threshold_curve : full per-bin curve (used for beacon detection and plotting)
        self.threshold       : mean of the curve, a single number (used to colour the waterfall)
        """
        if self.dynamic_thershold:
            baseline = mf.local_mean_curve(self.max_hold_power, self.neighbors)
        else:
            baseline = np.zeros_like(self.max_hold_power, dtype=float)

        self.threshold_curve = baseline + self.starting_threshold
        self.threshold = float(np.mean(self.threshold_curve))

        #print(self.threshold)

    def get_beacons(self):
        #print(f"{self.max_hold_power[50]}")
        return mf.finde_beacons(self.max_hold_power, self.full_spectrum, self.threshold_curve)
        

    def get_current_center_freq(self):
        return self.SD.info["center_freq"]

    def get_percentage(self):
        return self.percentage

    def get_threshold(self):
        # single number: mean of the threshold curve
        return self.threshold

    def get_threshold_curve(self):
        return self.threshold_curve
        


    



if __name__ == "__main__":




    scnr = scanner(
        sample_rate=2.4e6,
        min_freq=88e6,
        max_freq=110e6,
        smothnes=10,
        gain=20,
        threshold=2,
        beacon_bandwidth=100e3,
        outer_cut = 400e3, 
        dc_spike_cut = 5e3,
        dynamic_thershold = True,
        neighbors = 100

    )

    N_scanns = 5    

    #for i in range(N_scanns):
        #print(f"{i}/{N_scanns}",end='')
        #water_fall_image = []

    scnr.scan()
    beacons = scnr.get_beacons()
    frame = scnr.get_frame()


    print(f"detected {len(beacons)} tansminion sources")


    freqs = frame["freqs"]
    power = frame["power"]

    #water_fall_image.append(frame)

    #utils.plot_spectrum(power, freqs, beacons)

    #utils.plot_waterfall(water_fall_image)

    scnr.close()