from os.path import exists
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torchaudio

df = pd.read_csv('./../Dataset/dataset.csv')
print(df.head())

# Select subdataset without southern_english_male, given that there are missing files 
df = df[~df['filename'].str.contains('./datasets/southern_english_male/')].copy()
 
 
def extract_audios_1d_array(df):

    all_data = []   
    cont = 0
    out = 0
    # Do it every thounsand files, instead of 13000 audios at the same time. Otherwise .npy file is too big
    for l in range(14):
        arrays = []
        if out == 1: 
            break 
        
        print('turn',cont,cont+1000)
        for i in range(cont,cont+1000):

            if i == 50:#len(df['filename']):
                out = 1
                break 

            if i%500 == 0:
                print(i)
            audio_path ="./../.." + df['filename'].iloc[i][1:]

            # if exists(audio) == False:
            #     errors.append(df['filename'].iloc[i])
            #     print(audio,'------',df['filename'].iloc[i])

            try: 
                data, samplerate = sf.read(audio_path, dtype='float32')
                arrays.append(data)
                all_data.append(data)
            except:
                print('error: ',df['filename'].iloc[i]) 

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
    for i in range(len(df['filename'])):
        if i%500 == 0:
            print(i)
        audio_path ="./../.." + df['filename'].iloc[i][1:]        
        try: 
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform.to(device)
            if sample_rate != bundle.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

            with torch.inference_mode():
                features, _ = model.extract_features(waveform)   
                            
            torch.save(features, './../../datasets/wave2vec_features/wave2vec_features'+str(i)+'.pt')
            all_data.append(features)
        except: 
            print('error: ',df['filename'].iloc[i])

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




# extract_wave2vec_features(df)

# save_labels_to_tensor(df)

data = extract_audios_1d_array(df)

print('data size: ', len(data))
padded_data = padding(data)
