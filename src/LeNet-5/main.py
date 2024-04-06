from functions import *
from data_loader import *

batch_size = 128
epochs = 10
learning_rate = 0.001
model = 'keras'

def main():
    # --- KERAS ---
    if model == 'keras':
        (train_set, train_labels), (test_set, test_labels) = LoadDataKeras()
        model_keras = LeNet5Keras(train_set, train_labels, test_set, test_labels, epochs, batch_size, learning_rate)
        history = model_keras.TrainModel()
        DisplayStats((epochs, batch_size, learning_rate), history.history)

        print('Showing test samples')
        for sample in test_set:
            DisplaySample(model, model_keras, sample)

    # --- TORCH ---
    else:
        (trainloader, testloader) = LoadDataTorch(batch_size)
        model_torch = LeNet5Torch(trainloader, testloader, epochs, batch_size, learning_rate)
        history = TrainModel(model_torch, torch.optim.Adam(model_torch.parameters(), lr=learning_rate), nn.CrossEntropyLoss())
        DisplayStats((epochs, batch_size, learning_rate), history)

        print('Showing test samples')
        for idx, data in enumerate(testloader):
            samples, _ = data
            for sample in samples:
                DisplaySample(model, model_torch, sample)
    return

main()