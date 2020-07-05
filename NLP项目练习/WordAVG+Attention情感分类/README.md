通过torchtext数据预处理  
定义WordAVGModel  
引入Attention加权平均机制，cosine_similarity计算attention  
二分类任务BCEWithLogitsLoss计算损失，通过Adam算法优化参数  
模型训练过程引入Mask来对padding的数据进行处理  
