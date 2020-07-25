import torch
import time

from ._base_trainer import BaseTrainerMixin
from ..utils.common import color_print, ctqdm, set_seed
from ..utils.metrics import accuracy_score, classification_report


class ATCTrainer(BaseTrainerMixin):
    def __init__(self, device=None, batch_size=64, epoch_num=5):
        super(ATCTrainer, self).__init__()
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.device = device

    def train(self, model, train_data, dev_data, optimizer, scheduler,
              summary_writer, verbose=1, label2idx=None):
        model.to(self.device)
        for epoch in range(self.epoch_num):
            start_time = time.time()

            model.train()
            train_loss = 0
            avg_train_loss = 0
            for i, train in enumerate(train_data):
                train_input_ids = train[0].to(self.device)
                train_labels = train[1].to(self.device)
                train_mask = None if len(train) < 3 else train[2].to(self.device)

                optimizer.zero_grad()
                logits, loss = model(inputs=train_input_ids, labels=train_labels, mask=train_mask)

                train_loss += loss.item()
                avg_train_loss = train_loss / (i + 1)

                summary_writer.add_scalar(title="loss", value=loss.item(), n_iter=None)
                summary_writer.add_scalar(title="lr", value=scheduler.get_lr()[0], n_iter=None)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                color_print("[Epoch] {:0>3d}".format(epoch),
                            "[Batch] {:0>5d}".format(i),
                            "[lr] {:0>.6f}".format(scheduler.get_lr()[0]),
                            "[avg train loss] {:0>.4f}".format(avg_train_loss),
                            "[time] {:<.0f}s".format(time.time() - start_time),
                            verbose=verbose)

            # on-time evaluate model
            model.eval()
            test_loss = 0
            avg_test_loss = 0
            pred_labels, test_labels = [], []
            for i, test in enumerate(dev_data):
                test_input_ids = test[0].to(self.device)
                test_label = test[1].to(self.device)
                test_mask = None if len(test) < 3 else test[2].to(self.device)

                with torch.no_grad():
                    pred_label, loss = model(inputs=test_input_ids, labels=test_label, mask=test_mask)
                    pred_labels.append(torch.argmax(pred_label.cpu(), -1).float())
                    test_labels.append(torch.argmax(test_label.cpu(), -1))
                    test_loss += loss
                    avg_test_loss = test_loss / (i + 1)

            color_print("[Epoch] {:0>3d}".format(epoch),
                        "[lr] {:0>.6f}".format(scheduler.get_lr()[0]),
                        "[avg train loss] {:0>.4f}".format(avg_train_loss),
                        "[avg dev loss] {:>0.4f}".format(avg_test_loss),
                        "[time] {:<.0f}s".format(time.time() - start_time),
                        verbose=verbose + 1 if verbose != 0 else 0)

            if epoch == self.epoch_num - 1:
                acc = accuracy_score(torch.cat(pred_labels, dim=-1).numpy(),
                                     torch.cat(test_labels, dim=-1).numpy())
                color_print("The model test accuracy is: {:.5}".format(acc),
                            verbose=verbose + 1 if verbose != 0 else 0)

                label_sorted = sorted(label2idx.items(), key=lambda x: x[1])
                target_names = [i[0] for i in label_sorted]
                idx2label = {v: k for k, v in label2idx.items()}

                report = classification_report(
                    [idx2label[index] for index in torch.cat(test_labels, dim=-1).numpy()],
                    [idx2label[index] for index in torch.cat(pred_labels, dim=-1).numpy()],
                    target_names=target_names)
                color_print(report, verbose=verbose + 1 if verbose != 0 else 0)

    def predict(self, model, test_data, summary_writer, verbose=1, label2idx=None):
        model.to(self.device)
        model.eval()
        pred_labels = []
        for i, test in ctqdm(iterable=enumerate(test_data), desc="Batch test",
                             verbose=verbose, total=len(test_data)):
            test_input_ids = test[0].to(self.device)
            test_mask = None if len(test) < 2 else test[1].to(self.device)

            with torch.no_grad():
                pred_label, loss = model(inputs=test_input_ids, test_mask=test_mask)
                pred_labels.append(torch.argmax(pred_label.cpu().detach(), -1).float())
        idx2label = {v: k for k, v in label2idx.items()}
        ret = [idx2label[index] for index in torch.cat(pred_labels, dim=-1).numpy()]
        return ret
