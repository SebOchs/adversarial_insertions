import sys
import os
# Slurm fix
sys.path.append(os.getcwd())
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from student_lab.lit_Model import LitBERT, LitT5



def training(data_set_name, training_set, test_set, mode, batch_size=8, lr=0.00002, precision=False, accumulate_grad=8,
             ddp=False, val=''):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_macro",
        mode="max",
        filepath="models/" + data_set_name + '_' + mode + '_' + '{epoch}-{val_macro:.4f}',
        save_top_k=3
    )
    if mode == 'bert':
        model = LitBERT(training_set, test_set, batch_size, lr, val=val)
    if mode == 'T5':
        model = LitT5(training_set, test_set, batch_size, val=val)
    trainer = pl.Trainer(gpus=1, max_epochs=8, checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=100,
                         accumulate_grad_batches=accumulate_grad, check_val_every_n_epoch=1, num_sanity_val_steps=0)
    if precision:
        trainer = pl.Trainer(gpus=1, max_epochs=16, checkpoint_callback=checkpoint_callback,
                             progress_bar_refresh_rate=100, accumulate_grad_batches=accumulate_grad,
                             check_val_every_n_epoch=1, num_sanity_val_steps=1, precision=16, amp_level='O2',)
    if ddp:
        trainer = pl.Trainer(gpus=2, max_epochs=16, checkpoint_callback=checkpoint_callback,
                             progress_bar_refresh_rate=100, accumulate_grad_batches=accumulate_grad,
                             check_val_every_n_epoch=1, num_sanity_val_steps=0, num_nodes=1, distributed_backend='ddp')
    if ddp and precision:
        trainer = pl.Trainer(gpus=2, max_epochs=8, checkpoint_callback=checkpoint_callback,
                             progress_bar_refresh_rate=100, accumulate_grad_batches=accumulate_grad,
                             check_val_every_n_epoch=1, num_sanity_val_steps=1, num_nodes=1, distributed_backend='ddp',
                             precision=16, amp_level='O2')



    trainer.fit(model)
    print("finished training")


# SEB Training
# mnli
training("mnli", "datasets/preprocessed/bert/MNLI/train.npy", "datasets/preprocessed/bert/MNLI/dev_mm.npy", 'bert',
         batch_size=64, precision=True, accumulate_grad=1, ddp=True, val="datasets/preprocessed/bert/MNLI/dev_m.npy")
training("mnli", "datasets/preprocessed/T5/MNLI/train.npy", "datasets/preprocessed/T5/MNLI/dev_mm.npy", 'T5',
         batch_size=32, precision=True, accumulate_grad=2, ddp=True, val="datasets/preprocessed/T5/MNLI/dev_m.npy")
"""
# T5
training("seb", "datasets/preprocessed/T5/seb/train.npy", "datasets/preprocessed/T5/seb/test_ua.npy", 'T5',
         batch_size=32, precision=True, accumulate_grad=1)
"""

