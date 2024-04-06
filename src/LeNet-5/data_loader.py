from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def LoadDataKeras():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Data normalization
    x_train = x_train.astype('float32') / 255
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Data augmentation
    # datagen = ImageDataGenerator(
    #         featurewise_center=False,  # set input mean to 0 over the dataset
    #         samplewise_center=False,  # set each sample mean to 0
    #         featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #         samplewise_std_normalization=False,  # divide each input by its std
    #         zca_whitening=False,  # apply ZCA whitening
    #         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    #         zoom_range = 0.1, # Randomly zoom image 
    #         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #         horizontal_flip=False,  # randomly flip images
    #         vertical_flip=False)  # randomly flip images
    # datagen.fit(train_set)

    return (x_train, y_train), (x_test, y_test)

def LoadDataTorch(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))]
        )
    trainset = FashionMNIST(root='./src/LeNet-5/data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = FashionMNIST(root='./src/LeNet-5/data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return (trainloader, testloader)
