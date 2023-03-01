# 1 summary

https://sthalles.github.io/simple-self-supervised-learning/

### 名解

- triplet三元组：三元组是指一组三个输入样本，用于训练模型以学习嵌入。具体来说，三元组由**锚样本、正样本和负样本**组成。anchor样本是参考输入，正样本是与anchor属于同一类或类别的样本，而负样本是与anchor属于不同类或类别的样本。

  旨在学习一个函数，将每个input映射为embedding向量，

  使锚~正 距离最小，锚~负距离最大，triplet loss用来计算这两种损失

  在训练期间，模型会呈现一系列三元组，并为每个三元组计算三元组损失。目标是优化模型参数以最小化整个训练集的三元组损失。通过这样做，该模型学习生成嵌入，**这些嵌入对于相似的样本靠得很近，而对于不同的样本则相距很远。**

- latent space潜在空间：捕获原始高维数据的重要和相关特征的**低维空间，**是输入数据的**压缩表示**，可用于各种下游任务。在对比学习中，它指数据的已经被学习到的特征，可应用在计算对比损失上了

- ablation study消融研究：用于通过移除或修改模型的组件来研究模型性能的技术，以了解每个组件对模型整体性能的贡献。消融研究要求模型表现出“优雅的退化”，这意味着即使某些组件被移除或退化，模型也应该继续运行。

- optimizer优化算法：**调整NN的权重以minLoss**

  - ADAM：自适应lr优化
  - SGD：经典，固定lr，但需要更多的超参调整和更慢的收敛

- scheduler调度器：是一种在训练过程中**更新optimizer学习率lr**的算法。

  - CosineAnnealingLR余弦退火：以非常大的学习率开始，然后在下次增加学习率之前将其快速降低到接近 0 的值。



### 对比学习介绍

对比学习的核心是噪声对比估计（NCE）损失。

![image-20230301144920398](https://tkmfpicgo.oss-cn-hangzhou.aliyuncs.com/img/image-20230301144920398.png) 

> - **x+**是输入数据x的基准点，x+与x相关，且两者（x,x+）为一个正样本对。
>
>   通常x+是x的变换结果，变换可能有尺寸切割、旋转、种种数据增强等
>
>   :star:**强数据增强**对无监督学习非常有用，作者推荐随机裁剪、水平翻转、色彩抖动、高斯模糊
>
> - 而**x-**是与x不相近的样本，（x,x-）组成一个负样本对且不相关
>
> - **每个正对都有k个负对**
>
>   :star:试验结果表明**需要大量负对**来保证效果
>
> - NCE loss旨在放大正对与负对的差别
>
>   - **sim（）**函数是相似性（距离）度量，负责最小化正对之间的距离，最大化正对与负对的距离
>
>     通常它的形式是**余弦相似性**或**点积**
>
>   - **g（）**函数是一个CNNencoder（resnet50），最近的对比学习架构会用孪生网络学习正对与负对的embedding，之后将其运去算对比损失

简单来说，我们可以将对比任务视为用计算距离在一堆负例中找出正例。

### simCLR

该方法在自监督和半监督学习基准测试中达到了 SOTA

> state-of-the-art，SOTA DNN是可用于任何特定任务的**最佳模型**。如果 DNN 在性能准确度上得分很高（大约 90%-95%），则它被标记为 SOTA 模型。

SimCLR 使用对比学习来最大化同一图像的 2 个增强版本之间的一致性。

##### 构建

1. 给定一个输入原图，应用两个不同的数据增强手段创建两个**增强副本图片**

2. 将所有副本图片装入batch

3. :star2:因为要最大化负对的数量，所以将一个batch内的一张图与其他图进行组合成负对（不能与自身or另一个副本组合）

   若batch大小为N，则每张图都能组成（N-1）个负对

4. 用resnet-50作为convnet主干**f（）**，接收形状为**(224,224,3)**的**增强图像**并输出 2048 维embedding向量**H**

5. 将h输入到**投影头g（）**——由两个dense layers（2048个元的全连接层）组成的一个MLP，隐层有非线性激活函数relu，

   得到**向量Z**——潜在空间

   > model.add(Dense(32, input_dim=2048))
   >
   > MLP是一种全连接的前馈人工神经网络，输入——若干隐层——输出

   <img src="https://tkmfpicgo.oss-cn-hangzhou.aliyuncs.com/img/image-20230301193707676.png" alt="image-20230301193707676" style="zoom:80%;" /> 

6. 用对比损失函数

   - 先用**余弦相似度**（ 2 个非零向量之间夹角的余弦值）测量正对与负对间的值
   - 有了**相似矩阵**之后，我们执行一个**softmax**来得到整个模型的概率分布
   - 目的是使softmax分布

   

7. 训练结束后，丢弃投影头g，直接用主干得到的H投入下游任务



### 模型性能评估

##### sth

仅准确性Accuracy（正确数/总数）可能不足以评估模型的性能。根据具体任务以及考虑

1. 假阳性和假阴性之间的平衡

   > 重要的是要考虑具体任务以及每种错误类型的相关成本或风险。
   >
   > 在某些情况下，假阳性可能比假阴性成本更高或风险更大，例如在医疗诊断任务中，假阴性（实际患有疾病但被错误诊断为健康的患者）可能比假阳性（健康患者被诊断患有疾病）成本更高或风险更大。
   >
   > 在误报和漏报之间取得平衡的一种方法是调整模型的决策阈值。 （为假阴和假阳调整权重）

2. 精度Precision（真阳/预阳）——多少错抓

3. 召回率Recall（真阳/总阳）——多少漏抓

4.  F1 分数——精度和召回率的调和平均数*harmonic mean*，越高越好

5. ROCreceiver operating characteristic

##### 线性评估协议

linear **evaluation** protocol用于评估训练好的模型 生成向量表征的水平

1. 从训练好的神经网络中提取学习到的表征。
2. 使用有标记的下游数据集，在学习的**表征之上训练线性分类器**。
3. 
4. 



1. 加载trained模型
2. 将预训练模型的最后一层替换为适合新任务的新层
3. 冻结除最后一层以外的所有图层（不改变模型权重）
4. 使用新数据在新任务上训练模型。由于大部分层都被冻结，因此训练过程将比从头开始训练新模型快得多。
5. **fine-tune**——解冻部分层，再次训练



这个想法是在 SimCLR 编码器的固定表征上训练线性分类器。为此，我们获取训练数据，将其传递给预训练的 SimCLR 模型，并存储输出表示。注意，此时我们不需要投影头G了。

然后使用这些固定表示来使用训练标签作为目标来训练逻辑回归模型。然后，我们可以测量测试精度，并将其用作特征质量的度量。

### simclr其它待读

- 在半监督基准上进行无监督对比特征学习的结果；

- 向投影头添加非线性层的实验和好处；

- 使用大batch size的实验和好处；

- 对比目标训练大型模型的结果；

- 使用多种更强的数据增强方法进行对比学习的消融研究；

- 归一化embedding在训练对比学习模型的好处；

  > 可用于计算余弦相似度（只与角度有关，向量大小无关）

# 2  实操

https://www.kaggle.com/code/aritrag/simclr

### 1数据增强+生成数据集

> contrastive_learning_dataset.py

```python
# s

```





```python
#k	
class CustomDataset(Dataset):

    def __init__(self, list_images, transform=None):
        """
        Args:
            list_images (list): List of all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.list_images = list_images
        self.transform = transform

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.list_images[idx]
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)

        return image
    
# 
```



### 2数据集导入

> 位于run.py中的main函数

pytorch中两类原始导入函数

- torch.utils.data.DataSet
- torch.utils.data.DataLoader

DataSet类 封装原始数据，DataLoader类 遍历创建的DataSet

```python
# s
	#👇数据集（压缩）
data='./datasets' #self.root_folder
	#ContrastiveLearningDataset中定义
dataset = ContrastiveLearningDataset(data)
	
    #读取torch自带数据集
def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True, transform=ContrastiveLearningViewGenerator( self.get_simclr_pipeline_transform(32), n_views), download=True),
    #⭐
train_dataset = dataset.get_dataset(dataset_name, n_views)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
     num_workers=workers,
    # num_workers=os.cpu_count(),
    pin_memory=True, 
    drop_last=True)
```

```python
# k
	#re_dataset
flowers_ds = CustomDataset(
    list_images=glob.glob("/kaggle/input/flowers-recognition/flowers/flowers/*/*.jpg"),
    transform=custom_transform
)

	#展示自定义数据集
plt.figure(figsize=(10,20))
def view_data(flowers, index):
    for i in range(1,6):
        images = flowers[index]
        view1, view2 = images
        plt.subplot(5,2,2*i-1)
        plt.imshow(view1.permute(1,2,0))
        plt.subplot(5,2,2*i)
        plt.imshow(view2.permute(1,2,0))

view_data(flowers_ds,2000)
	#re_dataloader
train_dl = torch.utils.data.DataLoader(
    flowers_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=os.cpu_count(),
    drop_last=True,
    pin_memory=True,
)
```

### 3 model

> resnet_simclr.py

```python
# s

self.criterion =torch.nn.CrossEntropyLoss().to(device)
features = self.model(images)
	#⭐info_nce_loss
logits, labels = self.info_nce_loss(features)
loss = self.criterion(logits, labels)

	#arch:可选resnet18/50
model = ResNetSimCLR(base_model=arch, out_dim=out_dim)
	#👇不用动
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
	#余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1)



    #  如果gpuindex是负数or不存在就会no-op（什么都不做）
    with torch.cuda.device(gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler)
        #👇见2
        simclr.train(train_loader)
```

```python
# k
	#👇不变
simclr_model = SimCLR().to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer=torch.optim.Adam(simclr_model.parameters())


for i, views in enumerate(train_dl):#👈遍历dataloader
        features = simclr_model([view.to(DEVICE) for view in views])
        
        logits, labels = cont_loss(features, temp=2)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
```

### 4 对比损失

```python
#s
   def info_nce_loss2(self, features):
        
        #【1】

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
```

```python
#k

 	#【1】
LABELS = torch.cat([torch.arange(BATCH_SIZE) for i in range(2)], dim=0)
LABELS = (LABELS.unsqueeze(0) == LABELS.unsqueeze(1)).float() # Creates a one-hot with broadcasting
LABELS = LABELS.to(DEVICE) #128,128

def cont_loss(features, temp):

    similarity_matrix = torch.matmul(features, features.T) # 128, 128
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(LABELS.shape[0], dtype=torch.bool).to(DEVICE)
    # ~mask is the negative of the mask
    # the view is required to bring the matrix back to shape
    labels = LABELS[~mask].view(LABELS.shape[0], -1) # 128, 127
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # 128, 127

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) # 128, 1

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) # 128, 126

    logits = torch.cat([positives, negatives], dim=1) # 128, 127
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE)

    logits = logits / temp
    return logits, labels
```

