import gc
import torch
import numpy as np
from glob import glob 

# Custom files 
from architecture.ae_ncp import ncp
from train_dry import model_fit
from test_dry import model_test


torch.manual_seed(7777)
np.random.seed(7777)

domain = ['source', 'target'] 
class_names = ['ToyTrain', 'gearbox', 'ToyCar', 'bearing', 'valve', 'fan', 'slider']

def main(config = None): 
    model = ncp()
    # model_name = "NCP_MSE_LOSS_2D_SINGLE_AE_32_4_ALL_BN_x3"
    # dataset_path = "./data/features/classes/test_sr_16e3_bearing_crop4_featuremfccADD_labelx3.pkl"

    model_fit(
        batch_size = 50,
        learning_rate = 1e-4,
        epoch = 3,
        dataset_path = "./data/features/classes/train_sr_16e3_valve_crop4_featuremfcc_s04.pkl",
        model = model,
        mode = "train",
    )

    # Clear gpu memory
    torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
        model_test(
            batch_size = 1000,
            dataset_path = "./data/features/classes/test_sr_16e3_valve_crop4_featuremfcc_s04.pkl",
            model = model,
        )

if __name__ == "__main__": 
    main()