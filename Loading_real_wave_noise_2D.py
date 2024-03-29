import os 
import torch
import torchaudio
import torchaudio.transforms as T
import librosa


def minmaxscaler(data):
    min = data.min()
    max = data.max()  
    return (data)/(max-min)


def resample_wav(waveform, sample_rate, resample_rate):
    resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)
    return resampled_waveform


class transforms_construction():
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512, n_mel=64, TwoD_nfft=256, TwoD_Hop=128):
        self.Sample_Rate = sample_rate
        self.N_FFT = n_fft
        self.Hop_Num = hop_length
        self.Mel_Num = n_mel
        self.TwoD_FFT = TwoD_nfft
        self.TwoD_Hop = TwoD_Hop
    
    def __transformation__(self, Type = 'Mel' ):
        if Type == 'Mel':
            transformation = torchaudio.transforms.MelSpectrogram(sample_rate=self.Sample_Rate, n_fft=self.N_FFT, hop_length=self.Hop_Num, n_mels=self.Mel_Num) # torch.Size([1, 64, 32])
        elif Type == 'Spec':
            transformation = torchaudio.transforms.Spectrogram(n_fft=self.TwoD_FFT, hop_length=self.TwoD_Hop, power=2, center=False, onesided=True) #torch.Size([1, 129, 124])
        else:
            transformation = None
        return transformation
    

def loading_real_wave_noise(folde_name, sound_name):
    SAMPLE_WAV_SPEECH_PATH = os.path.join(folde_name, sound_name)
    waveform, sample_rate = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)
    resample_rate = 16000
    waveform = resample_wav(waveform, sample_rate, resample_rate)
    return waveform, resample_rate


def waveform_to_spectorgram(waveform):
    waveform = minmaxscaler(waveform) # minmax normalization
    trasformation = transforms_construction().__transformation__(Type='Mel')
    spectorgram = trasformation(waveform)
    spectorgram = librosa.core.power_to_db(spectorgram) # convert to dB
    spectorgram = torch.from_numpy(spectorgram)
    return spectorgram