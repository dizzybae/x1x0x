import logging
import os
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

# 参数设置
torch.manual_seed(0)
batch_size=256
# device = torch.device('cpu')
gpu_index=0
disable_cuda=False
fp16_precision=True
device = torch.device('cuda')
temperature=0.07
n_views=2
epochs=200
arch='resnet18'
log_every_n_steps=100
workers=12
out_dim=128
lr=0.0003
weight_decay=1e-4


data='./datasets'
dataset_name='stl10'

class SimCLR(object):

    # def __init__(self, *args, **kwargs):
    #     self.args = kwargs['args']
    #     self.model = kwargs['model'].to(device)
    #     self.optimizer = kwargs['optimizer']
    #     self.scheduler = kwargs['scheduler']
    #     self.writer = SummaryWriter()
    #     logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
    #     self.criterion = torch.nn.CrossEntropyLoss().to(device)

    def __init__(self, *args, **kwargs):
        # self.args = kwargs['args']
        

        self.model = kwargs['model'].to(device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(device)

    def info_nce_loss2(self, features):

        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     n_views * batch_size, n_views * batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # slect and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / temperature
        # 使得相似度更加平滑，方便训练。
        return logits, labels

    def info_nce_loss(self, features):

        # 生成n_views个大小为batchsize的张量，并将它们组合
        labels = torch.cat([torch.arange(batch_size)
                           for i in range(n_views)], dim=0)
        # unsqueeze：扩展张量的形状，在给定的张量的指定维度上增加一维
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)
        # 将输入的所有图片的特征向量一维正则化
        features = F.normalize(features, dim=1)

        # matmul计算两个矩阵的乘积
        # 这里是算所有特征向量与变换后的向量的乘积(点乘?)
        
        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     n_views * batch_size, n_views * batch_size)
        # assert similarity_matrix.shape == labels.shape
        # discard the main diagonal from both: labels and similarities matrix

        # eye生成label张量行数的单位矩阵
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(
            device)
        # labels[~mask]意为取label中除mask为1的位置之外的其他元素
        # 即去除对角线
        # .view改变张量性状，当张量的某一维的值是-1时，
        # 表示该维的大小可以由其他维的大小和张量总元素个数自动计算得到。
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape


        # select and combine multiple positives
        # 。bool将张量中所有值变为0或1
        # 取自乘中所有非零数将其重塑
        pos_pairs = []
        neg_pairs = []
        for i in range(batch_size):
            start_idx = i * n_views
            end_idx = start_idx + n_views
            idx_a = torch.arange(start_idx, end_idx)[
                labels[start_idx] == labels[start_idx:end_idx]]
            idx_b = torch.arange(start_idx, end_idx)[
                labels[start_idx] != labels[start_idx:end_idx]]
            pos_pairs += [(features[i], features[j])
                        for i in idx_a for j in idx_a if i < j]
            neg_pairs += [(features[i], features[j]) for i in idx_a for j in idx_b]

        
        pos_logits = []
        for pair in pos_pairs:
            pos_logits.append(torch.matmul(self.model(
                pair[0]), pair[1].unsqueeze(1))/temperature)
        pos_logits = torch.cat(pos_logits, dim=0)

        
        neg_logits = []
        for pair in neg_pairs:
            neg_logits.append(torch.matmul(self.model(
                pair[0]), pair[1].unsqueeze(1))/temperature)
        neg_logits = torch.cat(neg_logits, dim=0)

        
        target = torch.arange(batch_size *
                              n_views, device=device)
        target = torch.cat([target]*len(pos_pairs), dim=0)

        # 组合正样本与负样本
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = target

        # logits = logits / temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=fp16_precision)

        # save config file
        # save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {epochs} epochs.")
        logging.info(f"Training with gpu: {disable_cuda}.")

        for epoch_counter in range(epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(device)

                with autocast(enabled=fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % log_every_n_steps == 0:
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
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epochs)
        save_checkpoint({
            'epoch': epochs,
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
def main():
    # args = parser.parse_args()
    # assert n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        gpu_index = -1

    dataset = ContrastiveLearningDataset(data)

    train_dataset = dataset.get_dataset(dataset_name, n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=arch, out_dim=out_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()