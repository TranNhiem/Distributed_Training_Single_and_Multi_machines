# TranRick 2022/08 
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""
Here is an example how to run on two nodes, 2 GPUs each:
.. code-block: bash
    # first node, Lightning launches two processes
    MASTER_ADDR=node01.cluster MASTER_PORT=1234 NODE_RANK=0 python train.py --trainer.gpus 2 --trainer.num_nodes 2 \
        --data.data-path ...
    # second node, Lightning launches two processes
    MASTER_ADDR=node02.cluster MASTER_PORT=1234 NODE_RANK=1 python train.py --trainer.gpus 2 --trainer.num_nodes 2 \
        --data.data-path ...
"""

from dataloader import ImageNetDataModule
from model import ImageNetLightningModel 
#https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/utilities/cli.html
from pytorch_lightning.utilities.cli import LightningCLI

def main(): 
    # """Implementation of a configurable command line tool for pytorch-lightning."""
    cli= LightningCLI(
       description="Py-Lightning Distributed multi-node training", 
       model_class= ImageNetLightningModel,
       datamodule_class= ImageNetDataModule,
       seed_everything_default=123, 
       trainer_defaults=dict(accelerator="ddb", max_epoch=10)  
    )
    # TODO: determine per-process batch size given total batch size
    # TODO: enable evaluate
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

if __name__ == "__main__": 
    main() 

