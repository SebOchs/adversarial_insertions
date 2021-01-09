import pytorch_lightning as pl
from student_lab.dataloading import MyT5Dataset, MyBertDataset
from student_lab.lit_Model import LitBERT, LitT5
from torch.utils.data import DataLoader
import torch

# Check if warning message is true, wasn't
# model_test = torch.load('models/seb_bert_epoch=6-val_macro=0.7890.ckpt')
model = LitBERT.load_from_checkpoint('models/seb_bert_epoch=6-val_macro=0.7890.ckpt')
model.eval()
model.freeze()
trainer = pl.Trainer(gpus=1)
test_ua_dl = DataLoader(MyBertDataset("datasets/preprocessed/bert/seb/test_ua.npy"))
test_ud_dl = DataLoader(MyBertDataset("datasets/preprocessed/bert/seb/test_ud.npy"))
test_uq_dl = DataLoader(MyBertDataset("datasets/preprocessed/bert/seb/test_uq.npy"))
trainer.test(model, test_dataloaders=[test_ua_dl, test_ud_dl, test_uq_dl])