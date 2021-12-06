wandb = dict(
    project="cs236",
    entity="posegen",
)

device = "cuda"

# datasets
tesla_path_dataset = "/data/PoseGen_resized/cars/"
tesla_extension = "JPEG"

# stanford cars
stanford_cars_path_base = "/data/cars/car_devkit/devkit/"
stanford_cars_path_meta = f"{stanford_cars_path_base}/cars_meta.mat"
stanford_cars_path_train = f"{stanford_cars_path_base}/cars_train_annos.mat"
stanford_cars_path_test = f"{stanford_cars_path_base}/cars_test_annos_withlabels.mat"
stanford_cars_extension = "jpg"


seed_data = 0
n_pose_pairs = 1

# images
width, height = 256, 256

# training
seed = 0

# baselines
baselines_tesla_batch_size = 32
baselines_tesla_num_workers = 8

# transforms
transforms_mean_cars_tesla = (0.5, 0.5, 0.5)
transforms_std_cars_tesla = (0.24, 0.24, 0.24)
transforms_mean_poses_tesla = (0.25, 0.25, 0.25)
transforms_std_poses_tesla = (0.43, 0.43, 0.43)
# TODO: get these values
transforms_mean_cars_stanford_cars = (0.45, 0.45, 0.45)
transforms_std_cars_stanford_cars = (0.29, 0.29, 0.29)
transforms_mean_poses_stanford_cars = transforms_mean_poses_tesla
transforms_std_poses_stanford_cars = transforms_std_poses_tesla


# instance segmentation
instance_segmentation_threshold = 0.5
coco_instance_category_names = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
