import pytorch_lightning as pl
from student_lab.dataloading import MyT5Dataset, MyBertDataset
from student_lab.lit_Model import LitBERT, LitT5
from torch.utils.data import DataLoader
import torch


def testing(checkpoint, mode, test_dataloaders=[]):
    if mode == 'bert':
        model = LitBERT.load_from_checkpoint(checkpoint)
        if len(test_dataloaders) > 0:
            tests = []
            for test in test_dataloaders:
                tests.append(DataLoader(MyBertDataset(test)))
        else:
            tests = [model.test_dataloader()]
    if mode == 'T5':
        model = LitT5.load_from_checkpoint(checkpoint)
        if len(test_dataloaders) > 0:
            tests = []
            for test in test_dataloaders:
                tests.append(DataLoader(MyT5Dataset(test)))
        else:
            tests = [model.test_dataloader()]

    model.eval()
    model.freeze()
    trainer = pl.Trainer(gpus=1)
    trainer.test(model, test_dataloaders=tests)


testing('models/msrpc_bert_epoch=2-val_macro=0.8393.ckpt', 'bert')
