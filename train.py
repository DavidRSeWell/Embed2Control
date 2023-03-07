import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from tqdm import tqdm

class VAETrainer:
    """
    Train a VAE model on a given data set
    """
    def __init__(self, model, data, train_split: float, loss: callable, epochs: int, optim: callable, test_every: int, batch_size: int = 32) -> None:
       self.data = data
       self.epochs = epochs
       self.loss = loss
       self.model = model 
       self.optim = optim
       self.test_every = test_every


       train_data, test_data = torch.utils.data.random_split(data, [train_split, 1 - train_split])
       self._train_data, self._test_data = DataLoader(train_data, batch_size = batch_size), DataLoader(test_data, batch_size = batch_size)
        
    def test(self, model, data):
        model.eval()
        loss_test = 0.0
        with torch.no_grad():
            for b_idx, (x, y, _) in enumerate(data):
                x_hat = model(x)
                e = self.loss(x_hat, x).item()
                loss_test += abs(e)
        return loss_test / len(data)
            
    def train(self) -> tuple:
        test_errors = []
        train_errors = []
        for ep in tqdm(range(self.epochs)):
            if (ep % self.test_every == 0):
                test_error = self.test(self.model, self._test_data)
                test_errors.append(test_error)
                self.model.train()
            
            train_error = 0.0
            for b_idx, (xt, ut, xt_1) in enumerate(self._train_data):
                self.optim.zero_grad()
                x_hat = self.model(xt)
                e = self.loss(x_hat, xt)
                train_error += abs(e.item())
                e.backward()
                self.optim.step()
                
            train_errors.append(train_error / len(self._train_data))
                
        return test_errors, train_errors
        