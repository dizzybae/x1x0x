import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss2(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # slect and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        # ?????????????????????????????????????????????
        return logits, labels

    def info_nce_loss(self, features):

        # ??????n_views????????????batchsize??????????????????????????????
        labels = torch.cat([torch.arange(self.args.batch_size)
                           for i in range(self.args.n_views)], dim=0)
        # unsqueeze???????????????????????????????????????????????????????????????????????????
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        # ??????????????????????????????????????????????????????
        features = F.normalize(features, dim=1)

        # matmul???????????????????????????
        # ????????????????????????????????????????????????????????????(???????)
        # TODO:
        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape
        # discard the main diagonal from both: labels and similarities matrix

        # eye??????label???????????????????????????
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(
            self.args.device)
        # labels[~mask]?????????label??????mask???1??????????????????????????????
        # ??????????????????
        # .view???????????????????????????????????????????????????-1??????
        # ?????????????????????????????????????????????????????????????????????????????????????????????
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape


        # select and combine multiple positives
        # ???bool???????????????????????????0???1
        # ???????????????????????????????????????
        pos_pairs = []
        neg_pairs = []
        for i in range(self.args.batch_size):
            start_idx = i * self.args.n_views
            end_idx = start_idx + self.args.n_views
            idx_a = torch.arange(start_idx, end_idx)[
                labels[start_idx] == labels[start_idx:end_idx]]
            idx_b = torch.arange(start_idx, end_idx)[
                labels[start_idx] != labels[start_idx:end_idx]]
            pos_pairs += [(features[i], features[j])
                        for i in idx_a for j in idx_a if i < j]
            neg_pairs += [(features[i], features[j]) for i in idx_a for j in idx_b]

        # Compute the logits for the positive pairs
        pos_logits = []
        for pair in pos_pairs:
            pos_logits.append(torch.matmul(self.model(
                pair[0]), pair[1].unsqueeze(1))/self.args.temperature)
        pos_logits = torch.cat(pos_logits, dim=0)

        # Compute the logits for the negative pairs
        neg_logits = []
        for pair in neg_pairs:
            neg_logits.append(torch.matmul(self.model(
                pair[0]), pair[1].unsqueeze(1))/self.args.temperature)
        neg_logits = torch.cat(neg_logits, dim=0)

        
        target = torch.arange(self.args.batch_size *
                              self.args.n_views, device=self.args.device)
        target = torch.cat([target]*len(pos_pairs), dim=0)

        # ???????????????????????????
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = target

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
