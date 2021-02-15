import pytorch_lightning as pl
from student_lab.dataloading import MyT5Dataset, MyBertDataset
from student_lab.lit_Model import LitBERT, LitT5
from torch.utils.data import DataLoader
import torch


def testing(checkpoint, mode, test_dataloader=''):
    # test the performance of a model
    if mode == 'bert':
        model = LitBERT.load_from_checkpoint(checkpoint)
    if mode == 'T5':
        model = LitT5.load_from_checkpoint(checkpoint)


    model.eval()
    model.freeze()
    trainer = pl.Trainer(gpus=1)
    if len(test_dataloader) > 0:
        test = DataLoader(MyT5Dataset(test_dataloader))
        trainer.test(model, test_dataloaders=test)
    else:
        trainer.test(model)


testing('models/mnli_bert_epoch=1-val_macro=0.8304.ckpt', 'bert')
