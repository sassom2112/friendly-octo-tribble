# src/mnist.py
import torch
import torch.nn as nn
import torchvision.datasets
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import time, copy

# device config (train our model on GPU if it is available which is much faster)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# Transforms for dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# Load MNIST dataset
mnist_train = torchvision.datasets.MNIST('', train=True, transform =transform, download=True)
mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [50000, 10000])
mnist_test = torchvision.datasets.MNIST('', train=False, transform = transform, download=True)

# Define DataLoaders for train, validation, and test sets
batch_size = 100
dataloaders = {
    'train': DataLoader(mnist_train, batch_size=batch_size, shuffle=True),
    'val': DataLoader(mnist_val, batch_size=batch_size, shuffle=False),
    'test': DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
}

# Define dataset sizes
# Define dataset sizes
dataset_sizes = {
    'train': len(mnist_train),
    'val': len(mnist_val),
    'test': len(mnist_test)
}
print(f'dataset_sizes = {dataset_sizes}')

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    since = time.time()

    # Copy the initial model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    # Phases for training and validation
    phases = ['train', 'val']

    # Initialize dictionary to track training curves
    training_curves = {phase + '_loss': [] for phase in phases}
    training_curves.update({phase + '_acc': [] for phase in phases})

    # Iterate over epochs
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data in dataloaders for the phase
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.view(inputs.shape[0], -1)  # Flatten inputs
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Step the scheduler at the end of the training phase
            if phase == 'train':
                scheduler.step()

            # Calculate loss and accuracy for the epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            training_curves[phase + '_loss'].append(epoch_loss)
            training_curves[phase + '_acc'].append(epoch_acc)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it is the best on validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # Calculate total time elapsed during training
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model, training_curves

input_size = 28 * 28
hidden_size1 = 64
hidden_size2 = 64
hidden_size3 = 64
num_classes = 10

# External training parameters
batch_size = 10
learning_rate = 0.001
num_epochs = 50

# Two hidden layer classification model
class SimpleClassifier2Layer (nn.Module):
  def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
    super(SimpleClassifier2Layer, self).__init__()
    self.flatten = nn.Flatten()  # Add this line to flatten the input
    self.layers = nn.Sequential(
        nn.Linear(input_size, hidden_size1),
        nn.ReLU(),
        nn.Linear(hidden_size1, hidden_size2),
        nn.ReLU(),
        nn.Linear(hidden_size2, num_classes)
    )

  def forward(self, x):
    return self.layers(x)

# class SimpleClassifier3Layer (nn.Module):
#   def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
#     super(SimpleClassifier3Layer, self).__init__()
#     self.flatten = nn.Flatten()
#     self.layers = nn.Sequential(
#         nn.Linear(input_size, hidden_size1),
#         nn.ReLU(),
#         nn.Linear(hidden_size1, hidden_size2),
#         nn.ReLU(),
#         nn.Linear(hidden_size2, hidden_size3),
#         nn.ReLU(),
#         nn.Linear(hidden_size3, num_classes)
#     )

#   def forward(self, x):
#     return self.layers(x)
  
two_layer_model = SimpleClassifier2Layer(input_size, hidden_size1, hidden_size2, num_classes).to(device)
print(two_layer_model)

# three_layer_model = SimpleClassifier3Layer(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes).to(device)
# print(three_layer_model)

# Two-hidden-Layer Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(two_layer_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Train the model. We also will store the results of training to visualize
two_layer_model, two_layer_training_curves = train_model(
    two_layer_model,           # Model
    dataloaders,               # Dataloaders (train, val, test)
    dataset_sizes,             # Dataset sizes
    criterion,                 # Loss function (e.g., CrossEntropyLoss)
    optimizer,                 # Optimizer (e.g., Adam, SGD)
    scheduler,                 # Scheduler (optional)
    num_epochs=num_epochs          # Number of epochs
)

# Three-hidden-Layer Training
criterion = nn.CrossEntropyLoss() # CrossEntropyLoss for classification!
optimizer = torch.optim.Adam(three_layer_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Train the model. We also will store the results of training to visualize
# Train the model, passing dataloaders and optimizer in the correct order
three_layer_model, training_curves_three_layer = train_model(
    three_layer_model,
    dataloaders,        # dataloaders should come here, not the optimizer
    dataset_sizes,
    criterion,          # Criterion (loss function)
    optimizer,          # Optimizer should come here
    scheduler,          # Learning rate scheduler
    num_epochs=num_epochs
)

def plot_training_curves(training_curves, phases=['train', 'val'], metrics=['loss', 'acc']):
    epochs = list(range(len(training_curves['train_loss'])))

    for metric in metrics:
        plt.figure()
        plt.title(f"Training Curves - {metric}")

        for phase in phases:
            key = phase + '_' + metric
            if key in training_curves:
                data = training_curves[key]

                # Convert list of tensors to list of numpy arrays
                if isinstance(data, list) and isinstance(data[0], torch.Tensor):
                    data = [d.cpu().numpy() for d in data]  # Move each tensor to CPU and convert to numpy

                plt.plot(epochs, data, label=phase)

        plt.xlabel('epochs')
        plt.legend()
        plt.show()

def classify_predictions(model, device, dataloader):
    model.eval()
    all_labels = torch.tensor([]).to(device)
    all_scores = torch.tensor([]).to(device)
    all_preds = torch.tensor([]).to(device)

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = torch.softmax(model(inputs), dim=1)
        _, preds = torch.max(outputs, 1)
        scores = torch.max(outputs, 1)[0]
        all_labels = torch.cat((all_labels, labels), 0)
        all_scores = torch.cat((all_scores, scores), 0)
        all_preds = torch.cat((all_preds, preds), 0)

    return all_preds.detach().cpu(), all_scores.detach().cpu(), all_labels.detach().cpu()

def plot_metrics(model, device, dataloaders, phase='test'): 
    preds, scores, labels = classify_predictions(model, device, dataloaders[phase])

    # Confusion matrix for multi-class classification (MNIST has 10 classes)
    cm = metrics.confusion_matrix(labels, preds)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
    disp.plot()
    plt.title("Confusion Matrix - Counts")
    plt.show()

    # Normalize the confusion matrix
    ncm = metrics.confusion_matrix(labels, preds, normalize='true')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=ncm, display_labels=[str(i) for i in range(10)])
    disp.plot()
    plt.title("Confusion Matrix - Normalized Rates")
    plt.show()

    # Accuracy score
    accuracy = metrics.accuracy_score(labels, preds)
    print(f"Accuracy: {accuracy:.4f}")

    return cm

plot_training_curves(training_curves_three_layer, phases=['train', 'val'])

res = plot_metrics(three_layer_model, device, dataloaders, phase='test')
