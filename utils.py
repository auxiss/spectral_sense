import numpy as np



def freq_to_text(fr):

    if fr <= 1000:
        output = (f'{fr}Hz')
        
    elif 1000 < fr <= 1000000:
        fr = fr / 1000
        output = (f'{fr}kHz')
    elif 1000000 <= fr < 1000000000:
        fr = fr / 1000000
        output = (f'{fr}MHz')
    elif 1000000000 <= fr:
        fr = fr / 1000000000
        output = (f'{fr}GHz')

    return output

