############################################################

# Code for loading / extracting features from raw data 

############################################################

# import h5py
import librosa 
import numpy as np
import pickle as pkl

from tqdm import tqdm 
from glob import glob 

# Save list data 
def save_list(path, data):
    with open(path, 'wb') as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

# load list data 
def load_list(path): 
    with open(path, 'rb') as f:
        return pkl.load(f)

def main():
    is_test = True 
    dataset_type = "train"
    domain = ['source', 'target']
    crop_sec = 4
    padding = 1
    to_feature = "mfcc"
    # path = glob("../data/unziped/*")
    class_names = ['ToyTrain', 'gearbox', 'ToyCar', 'bearing', 'valve', 'fan', 'slider']

    if is_test == True: 
        dataset_type == "test"

    for index_numba in range(len(class_names)):
        print("#"*20, "Loading", class_names[index_numba], "#"*20)
        # Read path
        path = glob("../data/unziped/*" + class_names[index_numba] + "/" + dataset_type + "/*.wav")

        # Check domain 
        path = [i for i in path if i.split("_")[2] == domain[0]]

        # Check for empty list and things ...
        print("Found file counts :", len(path))

        if len(path) == 0: 
            print("#"*20, "ERROR", "#"*20)
            print("No file has been found")
            print("#"*20, "ERROR", "#"*20)
            return None

        print("First raw dataset path :", path[0])

        
        # Loading data [audio_wav, classes, domain]
        print("="*20, "Loading data", "="*20)
        raw_data = [[librosa.load(i, sr=16e3)[0], i.split("/")[3], i.split("_")[2]] for i in tqdm(path)]

        # label and domain encoding to int [audio_feature, label, source]
        audio_data = [[i[0], class_names.index(i[1]), domain.index(i[2])] for i in tqdm(raw_data)]

        # Cropping features to "crop_sec"
        print("="*20, "Cropping features", "="*20)
        audio_data = [[np.array(i[0][int(16e3) * 1:int(16e3) * crop_sec]), i[1], i[2]] for i in tqdm(audio_data)] 


        print("="* 20, "DEBUG", "="* 20)
        print(audio_data[0])
        print("="* 20, "DEBUG", "="* 20)

        # Extracting features
        print("="*20, "Extracting features", "="*20) 
        feature_data = [[librosa.feature.mfcc(y = i[0], sr = 16e3, n_mfcc=128,), i[1], i[2]] for i in tqdm(audio_data)] 

        # Saving data as .pkl format 
        print("="*20, "Saving raw audio", "="*20)
        save_list(
            "./data/features/classes/" + dataset_type + "_sr_16e3_" + class_names[index_numba] + "_crop" + str(crop_sec) + "_feature" + to_feature + ".pkl", 
            feature_data
        )

    return None

if __name__ == "__main__": main()