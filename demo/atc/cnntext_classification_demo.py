from lightnlp.core.atc import CNNTextClassification

model = CNNTextClassification(pre_trained_embed_path="../../data/embeddings/tencent_glove.6B.200d.txt",
                              epoch_num=1, lr=1e-3,
                              verbose=1, batch_size=64)
model.train(data_path="../../data/labelled_datasets/xtiny_atc_sensitive.txt",)

# model.visualize(name="loss")
# model.visualize(name="lr")

print("Model Structure:")
model.visualize(name="model")

pred_ret = model.predict(data=["如果随便填会出什么结果呢"],
                         data_path="../../data/labelled_datasets/xtiny_atc_sensitive.txt")

print(pred_ret)
