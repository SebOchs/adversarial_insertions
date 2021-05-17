import sys
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lit_Model import LitBERT, LitT5

# Slurm fix
sys.path.append(os.getcwd())


def training(data_set_name, training_set, test_set, mode, batch_size=8, lr=0.00002, precision=False, accumulate_grad=8,
             ddp=False, val='', labels=0):
    """
    training script for the models
    :param data_set_name: string / name of data set
    :param training_set: string / preprocessed train data
    :param test_set: string / preprocessed test set
    :param mode: string / model type
    :param batch_size: int / training batch size
    :param lr: float / learning rate for bert
    :param precision: boolean / add 16 bit precision to trainer module
    :param accumulate_grad: int / nr of gradients to accumulate
    :param ddp: boolean / enables training on multiple gpu's
    :param val: string / path to preprocessed val set, if it exists
    :param labels: int / nr of labels for berts classification layer
    :return: nothing
    """
    checkpoint_callback = ModelCheckpoint(
        monitor="val_macro",
        mode="max",
        filepath="models/" + data_set_name + '_' + mode + '_' + '{epoch}-{val_macro:.4f}',
        save_top_k=3
    )
    if mode == 'bert':
        model = LitBERT(training_set, test_set, batch_size, lr, val=val, num_labels=labels)
    if mode == 'T5':
        model = LitT5(training_set, test_set, batch_size, val=val)
    trainer = pl.Trainer(gpus=1, max_epochs=8, checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=100,
                         accumulate_grad_batches=accumulate_grad, check_val_every_n_epoch=1, num_sanity_val_steps=0)
    if precision:
        trainer = pl.Trainer(gpus=1, max_epochs=8, checkpoint_callback=checkpoint_callback,
                             progress_bar_refresh_rate=100, accumulate_grad_batches=accumulate_grad,
                             check_val_every_n_epoch=1, num_sanity_val_steps=0, precision=16, amp_level='O2', )
    if ddp:
        trainer = pl.Trainer(gpus=2, max_epochs=4, checkpoint_callback=checkpoint_callback,
                             progress_bar_refresh_rate=100, accumulate_grad_batches=accumulate_grad,
                             check_val_every_n_epoch=1, num_sanity_val_steps=0, num_nodes=1, distributed_backend='ddp')
    if ddp and precision:
        trainer = pl.Trainer(gpus=2, max_epochs=8, checkpoint_callback=checkpoint_callback,
                             progress_bar_refresh_rate=100, accumulate_grad_batches=accumulate_grad,
                             check_val_every_n_epoch=1, num_sanity_val_steps=0, num_nodes=1, distributed_backend='ddp',
                             precision=16, amp_level='O2')

    trainer.fit(model)
    print("finished training")


training("mnli", "datasets/preprocessed/T5/MNLI/train.npy", "datasets/preprocessed/T5/MNLI/dev_m.npy", 'T5',
         batch_size=8, accumulate_grad=4, ddp=True)
"""
training("wic", "datasets/preprocessed/T5/wic/train.npy", "datasets/preprocessed/T5/wic/dev.npy", "T5",
          batch_size=8, accumulate_grad=2)
"""
