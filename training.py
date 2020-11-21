import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lit_BERT import LitBERT

checkpoint_callback = ModelCheckpoint(
    monitor="val_macro",
    mode="max",
    filepath='models/albert_{epoch}-{val_macro:.4f}',
    save_top_k=3
)
test = LitBERT()
trainer = pl.Trainer(
    gpus=1,
    #num_nodes=1,
    #distributed_backend='ddp',
    max_epochs=16,
    checkpoint_callback=checkpoint_callback,
    accumulate_grad_batches=32
)

trainer.fit(test)

print("finished training")