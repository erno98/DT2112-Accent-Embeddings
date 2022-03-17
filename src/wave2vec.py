from os.path import exists
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torchaudio
import librosa
import re
import sox
 
def extract_audios_1d_array(df):

    all_data = []   
    cont = 0
    out = 0
    # Do it every thounsand files, instead of 13000 audios at the same time. Otherwise .npy file generated is too big
    for l in range(20):
        arrays = []
        if out == 1: 
            break 
        
        print('turn',cont,cont+1000)
        for i in range(cont,cont+1000):

            if i == len(df['filename']):
                out = 1
                break 

            # if i%500 == 0:
            print(i)
            audio_path ="./../.." + df['filename'].iloc[i][1:]

            # if exists(audio) == False:
            #     print(audio,'------',df['filename'].iloc[i])
 
            target_sr = 16000 # set it to None to have native sampling rate
            data, sr = librosa.load(audio_path, sr=None)
            # data = librosa.feature.melspectrogram(y=data, sr=sr)
            print(data.shape)
            arrays.append(data)
            all_data.append(data)

        # Convert to tensor and save
        print('saving---')
        with open('./../../datasets/data_array/audio_array'+str(i)+'.npy', 'wb') as f:
            np.save(f, np.array(arrays))
        
        
        cont += 1000
        
    return all_data


def extract_wave2vec_features(df):
    
    all_data = [] 

    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)

    # Go through each audio, convert it to array and extract its features. 
    for i in range( len(df['filename'])):
        print(i)
        audio_path ="./../.." + df['filename'].iloc[i][1:]        
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(device)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

        # Features is a list of tensors. Each tensor is the output of a transformer layer. 
        with torch.inference_mode():
            features, _ = model.extract_features(waveform)   

        # Instead of saving all the tensor, we can try to save only the last one               
        # torch.save(features[-1], './../../datasets/wave2vec_features/wave2vec_features'+str(i)+'.pt')
        all_data.append(features[-1])
    torch.save(all_data, './../../datasets/wave2vec_features/wave2vec_features'+str(i)+'.pt')

    return all_data


def save_labels_to_tensor(df):

    # Convert labels to numpy  and save it
    names=list(df.columns)
    for col in names:
        print(len(df[col].unique()),'values in ',col)
        print(df['accent'].unique())

    df['accent'] = df['accent'].astype('category')
    labels = pd.get_dummies(df['accent'])#, drop_first=True)

    print('\nlabels: \n',labels.head()) 
    print('size: ', labels.size )

    # Convert to tensor and save
    with open('./../Dataset/csv_features.npy', 'wb') as f:
        np.save(f, df.to_numpy())


def padding(data):

    '''given a batch, pad the data to the longest vector of the batch'''
    max = 0
    min = 10000000000
    pos_max = 0
    pos_min = 0
    for i in range(len(data)):
        if len(data[i]) > max:
            max = len(data[i])
            pos_max = i 
        if len(data[i]) < min:
            min = len(data[i])
            pos_min = i 
        
    print('max: ',max,', pos: ',pos_max)
    print('min: ',min,', pos: ',pos_min)

    '''Pad with zeros all the arrays to the largest one'''
    arrays_padded = np.zeros((len(data),max))
    max_pad = [0]*max
    for i in range(len(data)):
        if i%500 == 0:
            print(i)
        copy_array = max_pad.copy()
        copy_array[:len(data[i])] = data[i]
        copy_array = np.array(copy_array)
        arrays_padded[i] = copy_array

    return arrays_padded

def remove_characters(df):
    ''' remove special characters and change uppercases '''
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

    for i in range(len(df['line'])):
        df['line'].iloc[i] = re.sub(chars_to_ignore_regex, '', df['line'].iloc[i]).lower() + " "
        # Replace " " by "|", because models have "|" instead of " " as label
        df['line'].iloc[i] = re.sub(' ', '|', df['line'].iloc[i]) + " "

    return df




df = pd.read_csv('./../Dataset/dataset.csv')

# Select subdataset without southern_english_male, given that there are missing files 
df = df[~df['filename'].str.contains('./datasets/southern_english_male/')].copy()

# df = remove_characters(df)
 
# data = extract_audios_1d_array(df)
# extract_wave2vec_features(df)
 
# padded_data = padding(data)
