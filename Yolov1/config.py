ARCHITECTURE = [
    # Kernel size, out channels, stride, padding
    (7, 64, 2, "same"), # 3
    "max",
    (3, 192, 1, "same"), # 1
    "max",
    (1, 128, 1, "valid"), # 0
    (3, 256, 1, "same"), # 1
    (1, 256, 1, "valid"), # 0
    (3, 512, 1, "same"), # 1
    "max",
    (1, 256, 1, "valid"),
    (3, 512, 1, "same"), # 1
    (1, 256, 1, "valid"), # 0
    (3, 512, 1, "same"), # 1
    (1, 256, 1, "valid"), # 0
    (3, 512, 1, "same"), # 1
    (1, 256, 1, "valid"), # 0
    (3, 512, 1, "same"), # 1
    (1, 512, 1, "valid"), # 0
    (3, 1024, 1, "same"), # 1
    "max",
    (1, 512, 1, "valid"), # 0
    (3, 1024, 1, "same"), # 1
    (1, 512, 1, "valid"), # 0
    (3, 1024, 1, "same"), # 1
    (3, 1024, 1, "same"), # 1
    (3, 1024, 2, "same"), # 1
    (3, 1024, 1, "same"), # 1
    (3, 1024, 1, "same") # 1
]

# NETWORK PARAMS
GRID_SIZE = 7
NUM_BOXES = 2
NUM_CLASSES = 20
WIDTH = 448
HEIGHT = 448
BATCH_SIZE = 32
IN_CHANNELS = 3

# LOSS PARAMS
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5
OBJ_THRESHOLD = 0.7