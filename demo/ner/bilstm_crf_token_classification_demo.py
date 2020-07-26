from lightnlp.core.ner import BiLSTMCRFClassification


model = BiLSTMCRFClassification()
model.train(data_path="../data/labelled_datasets/xtiny_ner.txt",
            pre_trained_embed_path="../data/embeddings/tencent_glove.6B.200d.txt",
            epoch_num=2, lr=1e-3,
            verbose=1)