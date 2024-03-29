import pyaudio
import wave
import math
from datetime import datetime
from scipy import stats

from Loading_real_wave_noise_2D import loading_real_wave_noise
from Control_filter_selection import Control_filter_selection

# seconds: the duration of recorded noise
# you can set a threshold_db: when the sound is greater than this amplitude, start recording

class AudioRecorder:
    def __init__(self, seconds=1, chunk=1000, sample_format=pyaudio.paInt24, channels=1, fs=16000, input_device_index=1):
        self.seconds = seconds
        self.chunk = chunk
        self.sample_format = sample_format
        self.channels = channels
        self.fs = fs
        self.input_device_index = input_device_index
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.sample_format,
                        channels=self.channels,
                        rate=self.fs,
                        frames_per_buffer=self.chunk,
                        input=True,
                        input_device_index=self.input_device_index)


    def record(self, filename):
        # Start recording 1s noise
        frames = []
        for i in range(0, int(self.fs / self.chunk * self.seconds)):
            data = self.stream.read(self.chunk)
            frames.append(data)
        
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        # Load the recorded noise
        sound_name = 'output'
        waveform, resample_rate = loading_real_wave_noise(folde_name='', sound_name=sound_name+'.wav')
        
        # Predict control filter index using SFANC
        id_vector = Control_filter_selection(fs=16000, Primary_noise=waveform) # Primary_noise: torch.Size([1, XX])
        ID = id_vector[0]
        
        # Select the mode from control filters IDs
        mode = stats.mode(id_vector)
        ID = mode.mode[0]
        
        return ID