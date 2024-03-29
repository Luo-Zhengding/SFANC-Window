import torch

from Modified_ShufflenetV2 import Modified_ShufflenetV2
from Loading_real_wave_noise_2D import waveform_to_spectorgram


def load_weigth_for_model(model, pretrained_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location="cpu")
    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict[k]    
    model.load_state_dict(model_dict)


def minmaxscaler(data):
    min = data.min()
    max = data.max()    
    return (data)/(max-min)


def Casting_multiple_time_length_of_primary_noise(primary_noise, fs):
    assert  primary_noise.shape[0] == 1, 'The dimension of the primary noise should be [1 x samples] !!!'
    cast_len = primary_noise.shape[1] - primary_noise.shape[1]%fs
    return primary_noise[:,:cast_len] # make the length of primary_noise is an integer multiple of fs


#-------------------------------------------------------------
# Class : Control_filter_Index_predictor
#-------------------------------------------------------------
class Control_filter_Index_predictor():
    
    def __init__(self, MODEL_PATH, device, fs):
        
        self.device = device
        # set the model
        model = Modified_ShufflenetV2(num_classes=7)
        model = model.to(self.device)
        # loading coefficients
        load_weigth_for_model(model, MODEL_PATH)
        model.eval()
        
        
        self.model = model
        self.fs = fs
    
    def predic_ID(self, noise): # predict the noise index
        spectorgram = waveform_to_spectorgram(noise) # !!! 2D torch.Size([1, 64, 32])
        spectorgram = spectorgram.to(self.device)
        spectorgram = spectorgram.unsqueeze(0) # torch.Size([1, 1, 64, 32])
        prediction = self.model(spectorgram) # torch.Size([7])
        pred = torch.argmax(prediction).item()
        return pred
    
    def predic_ID_vector(self, primary_noise):
        # Checking the length of the primary noise.
        assert  primary_noise.shape[0] == 1, 'The dimension of the primary noise should be [1 x samples] !!!'
        assert  primary_noise.shape[1] % self.fs == 0, 'The length of the primary noise is not an integral multiple of fs.'
        
        # Computing how many seconds the primary noise contain.
        Time_len = int(primary_noise.shape[1]/self.fs)
        
        # Bulding the matric of the primary noise [times x 1 x fs]
        primary_noise_vectors = primary_noise.reshape(Time_len, self.fs).unsqueeze(1)
        
        # Implementing the noise classification for each frame whose length is 1 second. 
        ID_vector = []
        for ii in range(Time_len):
            ID_vector.append(self.predic_ID(primary_noise_vectors[ii]))
        return ID_vector


def Control_filter_selection(fs=16000, Primary_noise=None):
    
    # pretrained CNN model path
    MODEL_PATH = 'ShuffleNetV2_Synthetic.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    Pre_trained_control_filter_ID_pridector = Control_filter_Index_predictor(MODEL_PATH=MODEL_PATH, device=device, fs=fs)
    
    Primary_noise = Casting_multiple_time_length_of_primary_noise(Primary_noise, fs=fs)
    
    Id_vector = Pre_trained_control_filter_ID_pridector.predic_ID_vector(Primary_noise)
    
    return Id_vector