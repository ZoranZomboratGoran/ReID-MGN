import torch
from torch.optim import lr_scheduler

from opt import opt
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature

class App():
    def __init__(self, model, loss, data):
        self.data = data
        self.train_loader = data.train_loader
        self.train_val_loader = data.train_val_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        if opt.usecpu == False and torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model
        self.loss = loss
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)