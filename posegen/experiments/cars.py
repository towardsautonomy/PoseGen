from . import Config

experiments = [
    Config(
        dataset="stanford_cars",
        lambda_gan=1,
        lambda_full=0,
        lambda_object=1,
        lambda_background=0,
        pretrain=False,  # 1:1 with lambda_full?
        condition_on_object=True,
        condition_on_pose=True,
        condition_on_background=False,
        skip_connections=False,
    ),
]
