import os
import sys
import json
from data_utils import data_builder, CoNLLDataset
from BILSTM_CRF_model import BILSTM_CRF
from config import config


def train(config_path,continue_=False):
    #reading hyperparameters
    config_params = json.load(open(config_path))
    
    #loading hyperparams
    config_data = config(**config_params, load=False)
    #Creating data vocab.txt, chars.txt, tags.txt, and embeddings
    data_builder(config_data)
    
    #creating loading the data created earlier
    config_train = config(**config_params,load=True)
    
    #build model
    model = BILSTM_CRF(config_train)
    model.build()
    
    if continue_:
        try:
            model_path = config_params["model_path"]
            print("Loading weights from path:: ",model_path)
            model.restore_session(model_path)
            model.reinitialize_weights("proj")
            print("Restoring weights")
        except:
            print("Restoring weights failed")
            print("training from scratch")
            print(e)
            input()
    
    
    
    #data generators
    dev   = CoNLLDataset(config_train.train, config_train.process_words,
                         config_train.process_tags)
    train = CoNLLDataset(config_train.test, config_train.process_words,
                         config_train.process_tags)
    
    # train model
    model.train(train, dev)
    
    print("Trainig Complete!")
    print("Remove the events.tf files from the output directory if you don't need them. Note that removing them won't affect the predictions in anyway")

if __name__ == "__main__":
    print("Enter config json path")
    train(input())
