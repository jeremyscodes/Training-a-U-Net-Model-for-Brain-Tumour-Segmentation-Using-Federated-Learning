import flwr as fl
from model import Unet, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,trainloader,valloader,backbone, encoder_weights)->None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader
        #Instatiate the model that will be trained
        self.model = Unet(backbone, encoder_weights)
    
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        #parameters is a list of numpy arrays representing the weights of the global model
        # Copy parameters sent by the server into client's local model
        self.model.set_weights(parameters) #9:40 in video

        #lr = config['lr']
        #optim = config['optim']
        epochs = config['local_epochs']
        #BACKBONE = config['backbone']
        #weights = config['encoder_weights']
        #do local training
        self.model.fit(self.trainloader, self.valloader,epochs, batch_size=32,verbose=0)
        train(self.model, self.trainloader, self.valloader, epochs)
        print("from client fit: len(self.trainloader)=",len(self.trainloader))
        return self.model.get_weights(), len(self.trainloader), {} # for sending anything (like run time or metrics) to server

    def evaluate(self, parameters, config):
        # get global model to be evaluated on client's validation data

        self.model.set_weights(parameters)

        loss, iou, f1 = test(self.model, self.valloader)

        return float(loss), len(self.valloader), {'iou':iou, 'f1':f1}


def generate_client_fn(trainloaders, valloaders, backbone, encoder_weights):
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
        return FlowerClient(trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            backbone=backbone,
                            encoder_weights=encoder_weights,
                            )

    return client_fn