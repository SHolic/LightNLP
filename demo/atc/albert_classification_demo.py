from lightnlp.core.atc import AlbertClassification


model = AlbertClassification(pre_trained_model_path="../../data/albert_models/albert_tiny_bright/",
                             epoch_num=2,
                             batch_size=128,
                             lr=2e-5,
                             verbose=1)
model.train(data_path="../../data/labelled_datasets/xtiny_atc_sensitive.txt")

model.save("../../trained_models/albert_atc.bin")

model = AlbertClassification().load("../../trained_models/albert_atc.bin")

pred_ret = model.predict(data=["如果随便填会出什么结果呢"],
                          data_path="../../data/labelled_datasets/xtiny_atc_sensitive.txt",
                          batch_size=64, verbose=1)

print(pred_ret)

print("Model Structure:")
print(model.visualize("model"))
