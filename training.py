import sys
import os
# Slurm fix
sys.path.append(os.getcwd())
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lit_Model import LitBERT, LitT5



def training(data_set_name, training_set, test_set, mode, batch_size=8, lr=0.00002, precision=False, accumulate_grad=8,
             ddp=False, val='', labels=0):
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
                             check_val_every_n_epoch=1, num_sanity_val_steps=0, precision=16, amp_level='O2',)
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


training("rte", "datasets/preprocessed/T5/RTE/train.npy", "datasets/preprocessed/T5/RTE/dev.npy", 'T5',
         batch_size=8, accumulate_grad=1)
"""
training("wic", "datasets/preprocessed/T5/wic/train.npy", "datasets/preprocessed/T5/wic/dev.npy", "T5",
          batch_size=8, accumulate_grad=2)
"""