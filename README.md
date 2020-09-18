# LightNLP 

![](https://img.shields.io/badge/language-python3-blue.svg)
![](https://img.shields.io/badge/license-MIT-green.svg)
![](https://img.shields.io/badge/Coverage-10%-green.svg)


一个**轻量级**的**NLP**深度学习库，提供不同NLP任务建模的**简单**工具。

**LightNLP**的接口形式类似于**scikit-learn**，其目的在于可以通过**train**和**predict**方便快捷的训练较为复杂的NLP模型，为新手或项目时间不充裕的使用者，提供一个benchmark。

项目提供一些NLP的数据，如常用embedding，albert/bert pytorch预训练模型，训练数据等，放在**/data**文件夹下。

## 特点

LightNLP涉及的场景和模型有：

- 序列标注（NER）
  - bilstm
  - bilstm+crf
  - albert
  - abert+crf
  - cnn
- 本文分类（ATC）
  - bilstm
  - bilstm+attention
  - Han
  - ELMO
  - cnntext
  - rcnn
  - transformer
  - bert
  - albert
- 文本摘要（ATS）
- 文本生成（ATG）

## 开始

类似scikit-learn的使用方法，创建一个实例，调用**train**和**predict**函数实现模型的训练和预测，详细api和参数请见[文档]()。

#### 文本分类cnntext

```python
from lightnlp.core.atc import CNNTextClassification

model = CNNTextClassification(pre_trained_embed_path="pre_trained_embedding.txt", # 也可以不填
                              epoch_num=2, lr=1e-3, verbose=1)
# 训练
model.train(data_path="train_data.txt") # 句子与标签tab分隔，或者直接输入corpus和label

# 预测
pred_ret = model.predict(data=predict_data_list,       
                         data_path=predict_data_path)  # data和data_path选其一即可
print(pred_ret)

# 模型保存
model.save(path)
# 模型载入
model2 = CNNTextClassification().load(path)


model.visualize(name="loss") # 画出loss曲线
model.visualize(name="lr")   # 画出学习率曲线

print("Model Structure:")
model.visualize(name="model")  # 印出模型结构
```

#### 文本分类Albert

```python
from lightnlp.core.atc import AlbertClassification

model = AlbertClassification(pre_trained_model_path=pre_trained_model_path, # 必填，albert基于pytorch的预训练模型
            epoch_num=2, batch_size=128, lr=1e-5, verbose=1)

model.train(data_path=train_data_path)

model.save(path)

model2 = AlbertClassification().load(path)

pred_ret = model2.predict(data=predict_data_list,       
                          data_path=predict_data_path，
                          batch_size=64, verbose=1)

print(pred_ret)

print("Model Structure:")
print(model2.visualize("model"))
```


## 如何为此项目提供帮助



**LightNLP** is a extremely lightweight NLP libaray, it is a tool for building models promptly in different NLP scenes.

The interface of **LightNLP** likes **scikit-learn**, in order to implement 





This software is licensed with the MIT license. See LICENSE.txt for the full text.



序列标注（NER），本文分类（ATC），文本摘要（ATS）和文本生成（ATG）

