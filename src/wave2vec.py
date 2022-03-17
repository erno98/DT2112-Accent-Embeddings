from os.path import exists
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torchaudio
import librosa
import re
import matplotlib.pyplot as plt
# import sox
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
 
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


def compress_time(features):

    # print('show: ')
    # plt.imshow(features[0].cpu())
    features = features.numpy()[0]
    # print('size ft:', features.shape)
    
    # Take the average every 20 units of time
    feature = []
    features_small = []
    mean_list = []
    for j in range(features.shape[1]):
        for i in range(features.shape[0]):
            mean_list.append(features[i,j])
            # print(i, 'ft')
            # print('elemetn ',features[i,0])
            if (i%10==0) and (i!=0): 
                # I should have 30/40 samples per second
                # print(i)
                # print('the list is: ', mean_list)
                # print('its mean is: ',sum(mean_list) / len(mean_list) )
                feature.append(sum(mean_list) / len(mean_list))
                mean_list = []
        # Append the mean of the last values of the feature that didnt reach 20 elements
        if len(mean_list)!=0: #in case the length is a multiple of 20
            feature.append(sum(mean_list) / len(mean_list))
        features_small.append(feature)
        feature = []
    # print(np.array(features_small).shape)
    # plt.imshow(features_small)
    # plt.show()
    return np.array(features_small)

def compress_features(data, components = 100):
    print('initial shape: ',data.shape)
    #standarize the data onto unit scale (mean = 0 and variance = 1) 
    pca_data = data.T
    # pca_data = StandardScaler().fit_transform(pca_data)
    pca = PCA(n_components=components)
    reduced_data = pca.fit_transform(pca_data)  
    # print('expl variance: ', pca.explained_variance_)
    print('ratio: ', pca.explained_variance_ratio_)
    print('ratio sum: ', np.sum(pca.explained_variance_ratio_))
    print('shape: ',reduced_data.shape)
    return reduced_data


def extract_wave2vec_features(df):
    
    all_data = [] 

    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    bundle = torchaudio.pipelines.WAV2VEC2_BASE#WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)

    # Go through each audio, convert it to array and extract its features. 
    for i in range(len(df['filename'])):
        # print(i)
        '''choose the path where to take it from'''
        audio_path ="./../.." + df['filename'].iloc[i][1:]        
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(device)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

        # Features is a list of tensors. Each tensor is the output of a transformer layer. 
        with torch.inference_mode():
            features, _ = model.extract_features(waveform)   

        '''To compress the features and time'''
        # features = compress_time(features[-1])
        # features = compress_features(features)
        '''choose where to save it'''
        torch.save(features, './../../datasets/wave2vec_features/wave2vec_features'+str(i)+'.pt')
        print(features.shape)
    all_data.append(features)

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

    '''Pad the data (1d-array) to the longest vector of the batch'''
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
        
    # print('max: ',max,', pos: ',pos_max)
    # print('min: ',min,', pos: ',pos_min)

    '''Pad with zeros all the arrays to the largest one'''
    arrays_padded = np.zeros((len(data),max))
    max_pad = [0]*max
    for i in range(len(data)):
        # if i%500 == 0:
            # print(i)
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
df = remove_characters(df)
 
 '''generate 1d arrays from the audios'''
# data = extract_audios_1d_array(df)
'''pad the 1d arrays generated to the longest'''
# padded_data = padding(df)

'''given the audios,the function saves the features one by one as a tensor with axis [features, time] '''
data_features = extract_wave2vec_features(df)
 

