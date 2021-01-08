import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lit_Model import LitBERT, LitT5

checkpoint_callback = ModelCheckpoint(
    monitor="val_macro",
    mode="max",
    filepath='models/seb_t5_{epoch}-{val_macro:.4f}',
    save_top_k=3
)
test = LitT5("datasets/preprocessed/T5/seb/train.npy", "datasets/preprocessed/T5/seb/test_ua.npy")
trainer = pl.Trainer(
    gpus=1,
    # num_nodes=1,
    # distributed_backend='ddp',
    max_epochs=8,
    checkpoint_callback=checkpoint_callback,
    accumulate_grad_batches=4,
    precision=16,
    amp_level='O0',
    check_val_every_n_epoch=1,
    num_sanity_val_steps=0,
    progress_bar_refresh_rate=100
)
trainer.fit(test)

print("finished training")
