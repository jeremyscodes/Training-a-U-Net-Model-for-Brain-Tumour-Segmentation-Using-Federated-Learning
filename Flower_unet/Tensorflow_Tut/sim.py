import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import flwr as fl
import segmentation_models as sm
from keras.layers import Input, Conv2D
from keras.models import Model
from typing import Dict, List, Tuple
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from dataLoader import load_datasets

import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
parser = argparse.ArgumentParser(description="Flower Simulation with Tensorflow/Keras")

parser.add_argument(
    "--num_cpus",
    type=int,
    default=1,
    help="Number of CPUs to assign to a virtual client",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default=0.0,
    help="Ratio of GPU memory to assign to a virtual client",
)
class FlowerClient(fl.client.NumPyClient):
    def __init__(self,trainloader,valloader)->None:#backbone, encoder_weights
        super().__init__()
        print("FlowerClient init_______________")
        self.trainloader = trainloader
        self.valloader = valloader

        #Instatiate the model that will be trained
        self.model = get_model()#backbone, encoder_weights
    
    def get_parameters(self, config):
        print("get_parameters_______________") #This is where i need to convert the weights
        # Is this back from GPU
        # Is it numpy array
        #tensorflow command get weights mean, average of weights gpu
            #then check if weights trained before and after training
        #numpy average cpu
        return self.model.get_weights()

    def fit(self, parameters, config):
        print("fit_______________")
        #parameters is a list of numpy arrays representing the weights of the global model
        # Copy parameters sent by the server into client's local model
        self.model.set_weights(parameters) #9:40 in video

        #lr = config['lr']
        #optim = config['optim']
        epochs = config['local_epochs']
        #BACKBONE = config['backbone']
        #weights = config['encoder_weights']
        #do local training
        self.model.fit(self.trainloader, self.valloader, epochs=epochs, verbose=2,
           #callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./logs')]
           )
        print("from client fit: len(self.trainloader)=",len(self.trainloader))
        return self.model.get_weights(), len(self.trainloader), {} # for sending anything (like run time or metrics) to server

    def evaluate(self, parameters, config):
        print("evaluate_______________")
        # get global model to be evaluated on client's validation data

        self.model.set_weights(parameters)

        loss, iou, f1 = self.model.evaluate(self.valloader, batch_size=32, verbose=2)

        return float(loss), len(self.valloader), {'iou':iou, 'f1':f1}


def get_client_fn(trainloaders, valloaders):#, backbone, encoder_weights
    print("get client fn _______________")
    #to simulate clients
    '''Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    '''
    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        print(type(trainloader=trainloaders[int(cid)]))
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            )#backbone=backbone,encoder_weights=encoder_weights,

    return client_fn

def get_model(backbone, encoder_weights=None):
    print("get model_______________")
    preprocess_input = sm.get_preprocessing(backbone)
    base_model = sm.Unet(backbone, encoder_weights)
    
    inp = Input(shape=(None,None, 1))
    l1 = Conv2D(3, (1,1))(inp)
    out = base_model(l1)
    model = Model(inp, out, name=base_model.name)

    model.compile(
        'Adam',
        loss=sm.losses.dice_loss, # changed from bce_jaccard_loss
        metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)], #f1 is Dice score
    )
    return model

def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    print("weighted average_______________")
    """Aggregation function for (federated) evaluation metrics.

    It will aggregate those metrics returned by the client's evaluate() method.
    """
    # Multiply each metric of every client by the number of examples used
    iou_scores = [num_examples * m["iou_score"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metrics (weighted average)
    return {
        "iou_score": sum(iou_scores) / sum(examples),
        "f1_score": sum(f1_scores) / sum(examples)
    }


def get_evaluate_fn(testset):
    print("get_evaluated_fn_______________")

    """Return an evaluation function for server-side (i.e. centralised) evaluation."""
    x_test, y_test = next(iter(testset))
    print("x_test.shape",x_test.shape)
    print("y_test.shape",x_test.shape)


    # The `evaluate` function will be called after every round by the strategy
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar],):
        model = get_model()  # Construct the model
        print("About to set weights_________________")
        model.set_weights(parameters)  # Update model with the latest parameters
        print("Finished setting weights_______________")
        loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
        return loss, {"accuracy": accuracy}
    print("about to return get_evaluated_fn")
    return evaluate

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    # Parse input arguments
    args = parser.parse_args()
    
    # Create dataset partitions (needed if your dataset is not pre-partitioned)
    
    # 1. Load Data
    trainloaders, valloaders, testloader = load_datasets(cfg.num_clients, cfg.batch_size)
    print("Data Loaded")
    

    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,  # Sample 10% of available clients for training
        fraction_evaluate=1,  # Sample 5% of available clients for evaluation
        min_fit_clients=cfg.num_clients ,  # Never sample less than 10 clients for training
        min_evaluate_clients=cfg.num_clients ,  # Never sample less than 5 clients for evaluation
        min_available_clients=int(
            cfg.num_clients 
        ),  # Wait until at least n clients are available
        evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
        evaluate_fn=get_evaluate_fn(testloader),  # global evaluation function
    )

    # With a dictionary, you tell Flower's VirtualClientEngine that each
    # client needs exclusive access to these many resources in order to run
    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(trainloaders,valloaders),
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init
            # does nothing if `num_gpus` in client_resources is 0.0
        },
    )


if __name__ == "__main__":
    # Enable GPU growth in your main process
    enable_tf_gpu_growth()
    main()