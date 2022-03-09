import os
import utils
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import seed_everything


# load configurations

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', help='json config file path', default = 'config.yaml')
parser.add_argument('--eval_only', '-e', action='store_true',
                    help="evaluate trained model on validation data.")
parser.add_argument('--resume', '-r', action='store_true',
                    help="resume training from a given checkpoint.")
parser.add_argument('--test_run', action='store_true',
                    help="quick test run")
parser.add_argument('--job_identifier', '-j', help='Unique identifier for run,'
                                                    'avoids overwriting model.')
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

config = utils.load_json_config(args.config)

wandb_logger = WandbLogger(project='speech', config=config)