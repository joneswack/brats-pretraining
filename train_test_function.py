import torch
import torch.nn.functional as F

import os
import glob
from time import time
from datetime import datetime

from tensorboardX import SummaryWriter

import logging

def log(message):
    print(message)
    logging.info(message)

class ModelTrainer():
    
    def __init__(self, model_name, model, train_loader, val_loader, loss_fn, metric, lr=1e-3,
                 epochs=10, num_batches_per_epoch=10, num_validation_batches_per_epoch=3,
                 use_gpu=False, multi_class=False):
        super(ModelTrainer, self).__init__()
        
        self.use_gpu = use_gpu
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

        self.multi_class = multi_class
        
        # we should have one log dir per run
        # otherwise tensorboard will have overlapping graphs
        self.model_name = '{}_lr_{}_epochs_{}'.format(model_name, lr, epochs)
        self.log_dir = 'tensorboard_logs/{}/{}'.format(self.model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.save_dir = 'models/{1}_{0}'.format(self.model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.train_writer = SummaryWriter(self.log_dir + '/train')
        self.val_writer = SummaryWriter(self.log_dir + '/test')
        
    def run(self):
        t0 = time()
        
        # first val loss before training
        self.val_epoch(self.model, self.val_loader, 0)
        
        for epoch in range(1, self.epochs + 1):
            log('\n# Epoch {} #\n'.format(epoch))
            self.train_epoch(self.model, self.train_loader, self.optimizer, epoch)
            self.val_epoch(self.model, self.val_loader, epoch)

        time_elapsed = time() - t0
        log('\nTime elapsed: {:.2f} seconds'.format(time_elapsed))
        self.train_writer.close()
        self.val_writer.close()

        self.save_model(self.save_dir)
        
        return time_elapsed

    def train_epoch(self, model, train_loader, optimizer, epoch):
        model.train()
        train_loss = 0
        train_metric = [0, 0, 0]
        for batch_idx in range(self.num_batches_per_epoch):
            batch = next(train_loader)
            data = torch.from_numpy(batch['data'])
            target = torch.from_numpy(batch['seg']).type(torch.LongTensor)
            
            if self.multi_class:
                target_oh = torch.zeros(target.shape[0], 4, *target.shape[2:])
                target_oh.scatter_(1, target, 1)
                target = target_oh
            if self.use_gpu:
                data, target = data.cuda(), target.cuda()

            # data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            metric_output = self.metric(output, target)
            if self.multi_class:
                train_metric[0] += metric_output[0].item()
                train_metric[1] += metric_output[1].item()
                train_metric[2] += metric_output[2].item()
            else:
                train_metric[0] += self.metric(output, target).item()
            
            # loss before updating the weights (i.e. at the beginning of each iteration)
            iteration = (epoch-1) * self.num_batches_per_epoch + batch_idx
            self.train_writer.add_scalar('loss', loss, iteration)
            
        # for name, param in model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), iteration)
            
        train_loss /= self.num_batches_per_epoch

        metrics = ['edema', 'tumor_core', 'enhancing']
        for idx, m in enumerate(train_metric):
            m /= self.num_batches_per_epoch

            if not self.multi_class:
                metric_label = 'Dice'
            else:
                metric_label = metrics[idx]
            self.train_writer.add_scalar(metric_label, m, iteration)
            log('[Train] Avg. {}: {:.2f}'.format(metric_label, m))

        log('[Train] Avg. Loss: {:.2f}'.format(train_loss))

    def val_epoch(self, model, val_loader, epoch):
        model.eval()
        val_loss = 0
        val_metric = [0, 0, 0]
        
        with torch.no_grad():
            for batch_idx in range(self.num_validation_batches_per_epoch):
                batch = next(val_loader)
                data = torch.from_numpy(batch['data'])
                target = torch.from_numpy(batch['seg']).type(torch.LongTensor)
                
                if self.multi_class:
                    target_oh = torch.zeros(target.shape[0], 4, *target.shape[2:])
                    target_oh.scatter_(1, target, 1)
                    target = target_oh
                if self.use_gpu:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss += self.loss_fn(output, target).item()
                metric_output = self.metric(output, target)
                if self.multi_class:
                    val_metric[0] += metric_output[0].item()
                    val_metric[1] += metric_output[1].item()
                    val_metric[2] += metric_output[2].item()
                else:
                    val_metric[0] += self.metric(output, target).item()

        # iteration after processing all batches of the current epoch
        iteration = epoch * self.num_batches_per_epoch
        val_loss /= self.num_validation_batches_per_epoch
        metrics = ['edema', 'tumor_core', 'enhancing']
        for idx, m in enumerate(val_metric):
            m /= self.num_validation_batches_per_epoch

            if not self.multi_class:
                metric_label = 'Dice'
            else:
                metric_label = metrics[idx]
            self.val_writer.add_scalar(metric_label, m, iteration)
            log('[Val] Avg. {}: {:.2f}'.format(metric_label, m))
        
        self.val_writer.add_scalar('loss', val_loss, iteration)

        log('[Val] Avg. Loss: {:.2f}'.format(val_loss))
        
    def save_model(self, path):
        log('Saved to: {}'.format(path))
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()