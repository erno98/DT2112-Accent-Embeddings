from os.path import exists
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import torch
import torchaudio

df = pd.read_csv('./../Dataset/dataset.csv')
print(df.head())

# Select subdataset, given that there are errors with southern_english_male
df = df[~df['filename'].str.contains('./datasets/southern_english_male/')].copy()
# df = df[['filename','line', 'accent']]


 
 
def extract_audios_array(df):

    errors = []    
    cont = 0
    for l in range(14):
        arrays = []
        print('turno',cont,cont+1000)
        for i in range(cont,cont+1000):#len(df['filename'])):
            if i%500 == 0:
                print(i)
            audio ="./../.." + df['filename'].iloc[i][1:]

            # if exists(audio) == False:
            #     errors.append(df['filename'].iloc[i])
            #     print(audio,'------',df['filename'].iloc[i])
            
            try: 
                data, samplerate = sf.read(audio, dtype='float32')
                arrays.append(data)
            except: 
                errors.append(df['filename'].iloc[i])
        # Convert to tensor and save
        print('saving---')
        with open('./../Dataset/audio_features/audio_features'+str(i)+'.npy', 'wb') as f:
            np.save(f, np.array(arrays))
        
        cont += 1000





def extract_audios_features(df):
    arrays = []
    errors = []
    
    # Go through each audio, convert it to array and extract its features. 
    for i in range(len(df['filename'])):
        if i%500 == 0:
            print(i)
        audio ="./../.." + df['filename'].iloc[i][1:]        
        try: 
            waveform, sample_rate = torchaudio.load(audio)
            waveform = waveform.to(device)
            if sample_rate != bundle.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

            with torch.inference_mode():
                features, _ = model.extract_features(waveform)            
            # arrays.append(features)
        except: 
            errors.append(df['filename'].iloc[i])
            print('error')
        
        torch.save(features, './../Dataset/tensors/audio_features'+str(i)+'.pt')
        # with open('./../Dataset/tensors/audio_features'+str(i)+'.npy', 'wb') as f:
        #     np.save(f, np.array(features))
    return None


def save_labels(df):

    # Convert labels to numpy  and save it
    names=list(df.columns)
    for col in names:
        print(len(df[col].unique()),'values in ',col)
        print(df['accent'].unique())

    df['accent'] = df['accent'].astype('category')
    labels = pd.get_dummies(df['accent'])#, drop_first=True)

    print('\nlabels: \n',labels.head()) 
    print('size: ', labels.size )
    # for col in names:
    #   df[col] = df[col].astype('string')
    # df['accent'] = df['accent'].astype('bool')
    # df['accent'] = df['accent'].astype('int')

    # Convert to tensor and save
    with open('./../Dataset/csv_features.npy', 'wb') as f:
        np.save(f, df.to_numpy())

extract_audios_array(df)

# extract_audios_features(df)

# save_labels(df)