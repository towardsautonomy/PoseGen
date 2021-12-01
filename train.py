# import os
# import argparse
# from typing import Tuple
#
# import torch
# import torch.optim as optim
# import wandb
#
# from posegen.datasets import CarDataset
# from posegen.models import PoseGen_Discriminator, PoseGen
# from posegen.trainer import Trainer
#
#
# def parse_args():
#     r"""
#     Parses command line arguments.
#     """
#
#     root_dir = os.path.abspath(os.path.dirname(__file__))
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--dataset",
#         type=str,
#         default='StanfordCarsDataset',
#         choices=['StanfordCarsDataset', 'PoseGenCarsDataset'],
#         help="Dataset to use for training the model.",
#     )
#     parser.add_argument(
#         "--obj_data_dir",
#         type=str,
#         default=os.path.join(root_dir, "data"),
#         help="Path to object dataset directory.",
#     )
#     parser.add_argument(
#         "--bgnd_data_dir",
#         type=str,
#         default=os.path.join(root_dir, "data"),
#         help="Path to background dataset directory.",
#     )
#     parser.add_argument(
#         "--sil_data_dir",
#         type=str,
#         default=os.path.join(root_dir, "data"),
#         help="Path to silhouette dataset directory.",
#     )
#     parser.add_argument(
#         "--out_dir",
#         type=str,
#         default=os.path.join(root_dir, "out"),
#         help=(
#             "Path to output directory. "
#             "A new one will be created if the directory does not exist."
#         ),
#     )
#     parser.add_argument(
#         "--name",
#         type=str,
#         required=True,
#         help=(
#             "Name of the current experiment."
#             "Checkpoints will be stored in '{out_dir}/{name}/ckpt/'. "
#             "Logs will be stored in '{out_dir}/{name}/log/'. "
#             "If there are existing checkpoints in '{out_dir}/{name}/ckpt/', "
#             "training will resume from the last checkpoint."
#         ),
#     )
#     parser.add_argument(
#         "--resume",
#         default=False,
#         action="store_true",
#         help=(
#             "Resumes training using the last checkpoint in '{out_dir}/{name}/ckpt/' if set. "
#             "Throws error if '{out_dir}/{name}/' already exists by default."
#         ),
#     )
#     parser.add_argument(
#         "--seed", type=int, default=0, help="Manual seed for reproducibility."
#     )
#     parser.add_argument(
#         "--im_size",
#         type=int,
#         default=256,
#         help=(
#             "Images are resized to this resolution. "
#             "Models are automatically selected based on resolution."
#         ),
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=64,
#         help="Minibatch size used during training.",
#     )
#     parser.add_argument(
#         "--max_steps", type=int, default=150000, help="Number of steps to train for."
#     )
#     parser.add_argument(
#         "--repeat_d",
#         type=int,
#         default=1,
#         help="Number of discriminator updates before a generator update.",
#     )
#     parser.add_argument(
#         "--eval_every",
#         type=int,
#         default=500,
#         help="Number of steps between model evaluation.",
#     )
#     parser.add_argument(
#         "--ckpt_every",
#         type=int,
#         default=500,
#         help="Number of steps between checkpointing.",
#     )
#     parser.add_argument(
#         "--device",
#         type=str,
#         default=("cuda:0" if torch.cuda.is_available() else "cpu"),
#         help="Device to train on.",
#     )
#
#     return parser.parse_args()
#
#
# def train(
#     data_dir: str,
#     out_dir: str,
#     name: str,
#     resume: bool,
#     seed: int,
#     seed_data: int,
#     repeat_d: int,
#     max_steps: int,
#     eval_every: int,
#     ckpt_every: int,
#     device: str,
#     nz: int,
#     lr: float,
#     betas: Tuple[float, float],
#     im_size: int,
#     batch_size: int,
#     bgnd_data_dir: str,
#     sil_data_dir: str,
#     eval_split: int,
#     num_workers: int,
# ):
#     r"""
#     Configures and trains model.
#     """
#
#     # Print command line arguments and architectures
#     # pprint.pprint(vars(args))
#
#     # Setup dataset
#     if not os.path.exists(obj_data_dir):
#         raise FileNotFoundError(f"Data directory 'args.obj_data_dir' is not found.")
#
#     # Check existing experiment
#     exp_dir = os.path.join(out_dir, name)
#     if os.path.exists(exp_dir) and not resume:
#         raise FileExistsError(
#             f"Directory '{exp_dir}' already exists. "
#             "Set '--resume' if you wish to resume training or "
#             "change '--name' if you wish to start a new experiment."
#         )
#
#     # Setup output directories
#     log_dir = os.path.join(exp_dir, "log")
#     ckpt_dir = os.path.join(exp_dir, "ckpt")
#     for d in [out_dir, exp_dir, log_dir, ckpt_dir]:
#         if not os.path.exists(d):
#             os.mkdir(d)
#
#     # Fixed seed
#     torch.manual_seed(seed)
#
#     # Set parameters
#     # nz, lr, betas, eval_split, num_workers = (256, 2e-4, (0.0, 0.9), 0.1, 8)
#
#     # Setup models
#     # net_g = PoseGen_Generator()
#     net_g = PoseGen(nz=nz)
#     # net_d = StyleGAN2_Discriminator(c_dim=0, img_resolution=args.im_size, img_channels=3)
#     net_d = PoseGen_Discriminator()
#
#     # Configure optimizers
#     opt_g = optim.Adam(net_g.parameters(), lr, betas)
#     opt_d = optim.Adam(net_d.parameters(), lr, betas)
#
#     # Configure schedulers
#     sch_g = optim.lr_scheduler.LambdaLR(
#         opt_g, lr_lambda=lambda s: 1.0 - ((s * repeat_d) / max_steps)
#     )
#     sch_d = optim.lr_scheduler.LambdaLR(
#         opt_d, lr_lambda=lambda s: 1.0 - (s / max_steps)
#     )
#
#     ds_train = CarDataset()
#     ds_eval = ...
#     ds_test = ...
#     dl_train = ...
#     dl_eval = ...
#     dl_test = ...
#
#     # datasets = {
#     #     "StanfordCarsDataset": CarDataset,
#     #     "PoseGenCarsDataset": PoseGenCarsDataset,
#     # }
#     # dataset = datasets[dataset]
#     # dl_train, dl_eval, dl_test = get_dataloaders(
#     #     dataset, obj_data_dir, bgnd_data_dir, sil_data_dir,
#     #     im_size, batch_size, eval_split, num_workers
#     # )
#
#     # Configure trainer
#     trainer = Trainer(
#         net_g,
#         net_d,
#         opt_g,
#         opt_d,
#         sch_g,
#         sch_d,
#         dl_train,
#         dl_eval,
#         dl_test,
#         nz,
#         log_dir,
#         ckpt_dir,
#         torch.device(device),
#     )
#
#     # Train model
#     trainer.train(max_steps, repeat_d, eval_every, ckpt_every)
#
#
# # if __name__ == "__main__":
# #     train(parse_args())
