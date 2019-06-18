import torch
import torch.nn.functional as F

import os
import glob
from time import time
from datetime import datetime

from tensorboardX import SummaryWriter

class ModelTrainer():
    
    def __init__(self, model_name, model, train_loader, val_loader, loss_fn, metric, lr=1e-3,
                 epochs=10, num_batches_per_epoch=10, num_validation_batches_per_epoch=3, use_gpu=False):
        super(ModelTrainer, self).__init__()
        
        if use_gpu:
            model.cuda()
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.epochs = epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.num_validation_batches_per_epoch = num_validation_batches_per_epoch
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = loss_fn
        self.metric = metric
        
        # we should have one log dir per run
        # otherwise tensorboard will have overlapping graphs
        self.log_dir = 'tensorboard_logs/{}_lr_{}_epochs_{}/{}'.format(
            model_name, lr, epochs, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.train_writer = SummaryWriter(self.log_dir + '/train')
        self.val_writer = SummaryWriter(self.log_dir + '/val')
        
    def run(self):
        t0 = time()
        
        # first val loss before training
        self.val_epoch(self.model, self.val_loader, 0)
        
        for epoch in range(1, self.epochs + 1):
            print('\n# Epoch {} #\n'.format(epoch))
            self.train_epoch(self.model, self.train_loader, self.optimizer, epoch)
            self.val_epoch(self.model, self.val_loader, epoch)

        time_elapsed = time() - t0
        print('\nTime elapsed: {:.2f} seconds'.format(time_elapsed))
        self.train_writer.close()
        self.val_writer.close()
        
        return time_elapsed

    def train_epoch(self, model, train_loader, optimizer, epoch):
        model.train()
        train_loss = 0
        train_metric = 0
        for batch_idx in range(self.num_batches_per_epoch):
            batch = next(train_loader)
            data = torch.from_numpy(batch['data'])
            target = torch.from_numpy(batch['seg'])
            
            if self.use_gpu:
                data, target = data.cuda(), target.cuda()

            # data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_metric += self.metric(output, target).item()
            
            # loss before updating the weights (i.e. at the beginning of each iteration)
            iteration = (epoch-1) * self.num_batches_per_epoch + batch_idx
            self.train_writer.add_scalar('loss', loss / len(data), iteration)
            
        for name, param in model.named_parameters():
            self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), iteration)
            
        train_loss /= self.num_batches_per_epoch
        train_metric /= self.num_batches_per_epoch
        
        self.train_writer.add_scalar('metric', train_metric, iteration)

        print('[Train] Avg. Loss: {:.2f}, Avg. Metric: {:.2f}'.format(
            train_loss, train_metric))

    def val_epoch(self, model, val_loader, epoch):
        model.eval()
        val_loss = 0
        val_metric = 0
        
        with torch.no_grad():
            for batch_idx in range(self.num_validation_batches_per_epoch):
                batch = next(val_loader)
                data = torch.from_numpy(batch['data'])
                target = torch.from_numpy(batch['seg'])
                
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()
                # data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += self.loss_fn(output, target).item()
                val_metric += self.metric(output, target).item()

        val_loss /= self.num_validation_batches_per_epoch
        val_metric /= self.num_validation_batches_per_epoch
        
        # iteration after processing all batches of the current epoch
        iteration = epoch * self.num_batches_per_epoch
        self.val_writer.add_scalar('loss', val_loss, iteration)
        self.val_writer.add_scalar('metric', val_metric, iteration)

        print('[Val] Avg. Loss: {:.2f}, Avg. Metric: {:.2f}'.format(
            val_loss, val_metric))