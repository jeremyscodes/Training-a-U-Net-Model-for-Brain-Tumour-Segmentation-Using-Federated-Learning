from omegaconf import DictConfig
from model import Unet, test
def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):

        #if server_round > 50:
        #    lr = config.lr /10
        return {'lr':config.lr, 'local_epochs':config.local_epochs}
    
    return fit_config_fn

def get_evaluate_fn(backbone, encoder_weights, testloader):

    def evaluate_fn(server_round:int, parameters, config):
        model = Unet(backbone, encoder_weights)

        #possibly set device
        
        # use weights from server
        print("try set_weights")
        model.set_weights(parameters)
        print("Able to set_weights")
        #evaluate performance of global model on centralized dataset (testset)
        loss, iou, f1 = test(model,testloader)

        return loss, {'iou':iou, 'f1':f1}
    return evaluate_fn