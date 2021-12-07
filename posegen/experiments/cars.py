from . import Config


# options are "tesla" and "stanford_cars"
dataset = "tesla"

experiments = [
    # auto encoder
    Config(
        dataset=dataset,
        lambda_gan=1,
        lambda_full=1,
        lambda_object=1,
        lambda_background=0,
        pretrain=True,
        condition_on_object=False,
        condition_on_pose=False,
        condition_on_background=False,
        skip_connections=False,
    ),
    # pose only
    Config(
        dataset=dataset,
        lambda_gan=1,
        lambda_full=0,
        lambda_object=1,
        lambda_background=0,
        pretrain=False,
        condition_on_object=False,
        condition_on_pose=True,
        condition_on_background=False,
        skip_connections=False,
    ),
    # pose and object
    Config(
        dataset=dataset,
        lambda_gan=1,
        lambda_full=0,
        lambda_object=1,
        lambda_background=0,
        pretrain=False,
        condition_on_object=True,
        condition_on_pose=True,
        condition_on_background=False,
        skip_connections=False,
    ),
]
