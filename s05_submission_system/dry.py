import gc
import torch
import numpy as np
from glob import glob 

# Custom files 
from architecture.ae_ncp import ncp
from train_dry import model_fit
from test_dry import model_test

from torchsummary import summary


torch.manual_seed(7777)
np.random.seed(7777)

domain = ['source', 'target'] 
class_names = ['ToyTrain', 'gearbox', 'ToyCar', 'bearing', 'valve', 'fan', 'slider']

def main(config = None): 
    model = ncp()
    model = torch.load("./saved_model/eval/source/Scanner_0.8021150827407837.pkl")
    # model_name = "NCP_MSE_LOSS_2D_SINGLE_AE_32_4_ALL_BN_x3"
    dataset_path = "./data/features/stft/train_sr_16e3_valve_crop4_featurestft_s04.pkl"

    # summary(model, input_size=[(1, 256, 256), (1, 256, 256), (1, 256, 256)])

    model_fit( 
        batch_size = 8,
        learning_rate = 1e-3,
        epoch = 40,
        dataset_path = dataset_path,
        model = model,
        mode = "train",
    )

    # with torch.no_grad():
    #     model_test(
    #         batch_size = 100,
    #         dataset_path = dataset_path.replace("train", "test"),
    #         model = model,
    #     )

if __name__ == "__main__": 
    main()