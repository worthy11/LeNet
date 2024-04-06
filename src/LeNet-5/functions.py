import matplotlib.pyplot as plt
from lenet_keras import *
from lenet_torch import *

labels = {
    0: 'T-shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

def TrainModel(model, optimizer, loss_function):
    history: dict[str, list] = {
        'loss': list(),
        'accuracy': list()
    }
    
    for epoch in range(model.epochs):
        epoch_loss = 100.
        batch_loss = 0.

        model.train(True)
        for idx, data in enumerate(model.trainloader):
            samples, labels = data
            
            optimizer.zero_grad()
            outputs = model(samples)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            if idx % 100 == 99: # Average loss per 100 batches 
                batch_loss /= 100
                print('Avg loss in batches {}-{}: {}'.format(idx-99, idx+1, batch_loss))
                if batch_loss < epoch_loss:
                    epoch_loss = batch_loss
                batch_loss = 0.
        
        model.eval()
        with torch.no_grad():
            accuracy = 0.
            for idx, data in enumerate(model.testloader):
                samples, labels = data

                outputs = model(samples)
                for idx, out in enumerate(outputs):
                    out = list(out)
                    loss = loss_function(outputs, labels)
                    max_prob = max(out)
                    max_idx = out.index(max_prob)
                    accuracy += (max_idx == labels[idx])
            print('Accuracy in epoch {}: {}'.format(epoch+1, accuracy/(len(model.testloader)*model.batch_size)))
        history['loss'].append(epoch_loss)
        history['accuracy'].append(accuracy/(len(model.testloader)*model.batch_size))
    return history

def DisplayStats(params: tuple[int, int, float, float], history: dict):
    (epochs, batch_size, learning_rate) = params
    x = [_+1 for _ in range(epochs)]

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(x, history['loss'], marker='o')
    ax2.plot(x, history['accuracy'], marker='o')
    fig.legend([], loc='outside upper left', title='Batch size: {}\nLearning rate: {}\nOptimizer: Adam\nLoss: Cross Entropy'.format(batch_size, learning_rate))
    ax1.set_title('Loss')
    ax2.set_title('Accuracy')
    fig.suptitle('Training statistics per epoch\n')

    fig.tight_layout()
    plt.show()

def DisplaySample(type: str, model, sample) -> None:
    if type == 'keras':
        sample = np.expand_dims(sample, axis=0)
        label, confidence = model.Predict(sample)
        plt.imshow(sample[0, :, :, :])
    else:
        label, confidence = Predict(model, sample)
        plt.imshow(sample[0])
    plt.title('Label: {}\nConfidence: {}'.format(labels[label], confidence))
    plt.show()