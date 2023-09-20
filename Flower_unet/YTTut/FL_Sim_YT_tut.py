import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
import flwr as fl
import pickle
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
#from tqdm import tqdm, trange
from dataLoader import load_datasets
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    
    # 1. Load Data
    trainloaders, valloaders, testloader = load_datasets(cfg.num_clients, cfg.batch_size)
    print("Data Loaded")
    # Check the number of samples in the first dataset
    num_samples_first_dataset = sum(1 for _ in trainloaders[0])
    num_samples_first_dataset += sum(1 for _ in trainloaders[1])
    print(num_samples_first_dataset)

    # 2. Define clients
    #spawn client with client id x
    client_fn = generate_client_fn(trainloaders, valloaders, cfg.backbone,cfg.encoder_weights)

    # 3. Define your strategy
    strategy = fl.server.strategy.FedAvg(fraction_fit=0.0001,
                                         min_fit_clients=cfg.num_clients_per_round_fit, 
                                         fraction_evaluate=0.00001, #all clients are available
                                         min_evaluate_clients=cfg.num_clients_per_round_eval, # num clients used to evaluate global model
                                         min_available_clients=cfg.num_clients, #num clients participating at any time
                                         on_fit_config_fn=get_on_fit_config(cfg.config_fit),#look in server.py
                                         evaluate_fn=get_evaluate_fn(cfg.backbone,cfg.encoder_weights,testloader)#how server evaluates global model
                                         )
    # 4. Start Simulation
    history = fl.simulation.start_simulation(
          client_fn=client_fn,
          num_clients=cfg.num_clients,
          config = fl.server.ServerConfig(num_rounds=cfg.num_rounds), #how many rounds or federated learning
          strategy=strategy
    )
    print("About to save")

    # 6. Save results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / 'results.pkl'

    results = {'history': history}
    with open(str(results_path), 'wb') as h:
          pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    print("Finished Correctly")

if __name__ == "__main__":
        main()
