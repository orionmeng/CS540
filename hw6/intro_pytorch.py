import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    if training:
        dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    else:
        dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=training)
    return loader



def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128), # 128 nodes
        nn.ReLU(),
        nn.Linear(128, 64), # 64 nodes
        nn.ReLU(),
        nn.Linear(64, 10) # 10
    )
    return model



def train_model(model, train_loader, criterion, T):
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(T):
        correct = 0
        total = 0
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            opt.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Train Epoch: {epoch} Accuracy: {correct}/{total} ({accuracy:.2f}%) Loss: {running_loss/len(train_loader):.3f}")



def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    if show_loss:
        print(f"Average loss: {test_loss/len(test_loader):.4f}")
    print(f"Accuracy: {accuracy:.2f}%")



def predict_label(model, test_images, index):
    model.eval()
    with torch.no_grad():
        outputs = model(test_images)
        probabilities = F.softmax(outputs, dim=1)
        top_probabilities, top_indices = torch.topk(probabilities[index], 3)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
        for prob, idx in zip(top_probabilities, top_indices):
            print(f"{class_names[idx]}: {prob.item()*100:.2f}%")



if __name__ == '__main__':
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)

    model = build_model()
    print(model)

    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, criterion, 5)

    evaluate_model(model, test_loader, criterion, False)
    evaluate_model(model, test_loader, criterion, True)

    test_images = next(iter(test_loader))[0]
    predict_label(model, test_images, 1)
