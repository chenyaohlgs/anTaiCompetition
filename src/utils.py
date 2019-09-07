import torch
import os

def create_folder(fd):
    """ Create folders of a path if not exists
    Args:
        fd: str, path to the folder to create
    """
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def to_cuda_if_available(list_args):
    """ Transfer object (Module, Tensor) to GPU if GPU available
    Args:
        list_args: list, list of objects to put on cuda if available

    Returns:
        list
        Objects on GPU if GPUs available
    """
    if torch.cuda.is_available():
        for i in range(len(list_args)):
            list_args[i] = list_args[i].cuda()
    return list_args

class SaveBest:
    """ Callback of a model to store the best model based on a criterion
    Args:
        model: torch.nn.Module, the model which will be tracked
        val_comp: str, (Default value = "inf") "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
    Attributes:
        model: torch.nn.Module, the model which will be tracked
        val_comp: str, "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
        best_val: float, the best values of the model based on the criterion chosen
        best_epoch: int, the epoch when the model was the best
        current_epoch: int, the current epoch of the model
    """
    def __init__(self, val_comp="inf"):
        self.comp = val_comp
        if val_comp == "inf":
            self.best_val = np.inf
        elif val_comp == "sup":
            self.best_val = 0
        else:
            raise NotImplementedError("value comparison is only 'inf' or 'sup'")
        self.best_epoch = 0
        self.current_epoch = 0

    def apply(self, value):
        """ Apply the callback
        Args:
            value: float, the value of the metric followed
            model_path: str, the path where to store the model
            parameters: dict, the parameters to be saved by pytorch in the file model_path.
            If model_path is not None, parameters is not None, and the other way around.
        """
        decision = False
        if self.current_epoch == 0:
            decision = True
        if (self.comp == "inf" and value < self.best_val) or (self.comp == "sup" and value > self.best_val):
            self.best_epoch = self.current_epoch
            self.best_val = value
            decision = True
        self.current_epoch += 1
        return decision



class EarlyStopping:
    """ Callback of a model to store the best model based on a criterion
    Args:
        model: torch.nn.Module, the model which will be tracked
        patience: int, number of epochs with no improvement before stopping the model
        val_comp: str, (Default value = "inf") "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
    Attributes:
        model: torch.nn.Module, the model which will be tracked
        patience: int, number of epochs with no improvement before stopping the model
        val_comp: str, "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
        best_val: float, the best values of the model based on the criterion chosen
        best_epoch: int, the epoch when the model was the best
        current_epoch: int, the current epoch of the model
    """
    def __init__(self, model, patience, val_comp="inf"):
        self.model = model
        self.patience = patience
        self.val_comp = val_comp
        if val_comp == "inf":
            self.best_val = np.inf
        elif val_comp == "sup":
            self.best_val = 0
        else:
            raise NotImplementedError("value comparison is only 'inf' or 'sup'")
        self.current_epoch = 0
        self.best_epoch = 0

    def apply(self, value):
        """ Apply the callback

        Args:
            value: the value of the metric followed
        """
        current = False
        if self.val_comp == "inf":
            if value < self.best_val:
                current = True
        if self.val_comp == "sup":
            if value > self.best_val:
                current = True
        if current:
            self.best_val = value
            self.best_epoch = self.current_epoch
        elif self.current_epoch - self.best_epoch > self.patience:
            return True
        self.current_epoch += 1
        return False