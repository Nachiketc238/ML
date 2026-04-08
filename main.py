import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_loader import get_data_loaders
from model import LogisticRegression, SimpleNeuralNetwork, DeepMLP, SimpleCNN, ComplexCNN

# Training hyperparams
EPOCHS = 3
BATCH_SIZE = 64
DATASET = 'MNIST'

def train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS):
    model.train()
    for near_epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"    [Epoch {near_epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy

def main():
    print(f"==============================================")
    print(f"   5-MODEL COMPARISON ON {DATASET}             ")
    print(f"==============================================\n")

    train_loader, test_loader = get_data_loaders(dataset_name=DATASET, batch_size=BATCH_SIZE)
    
    models = {
        "Logistic Regression": LogisticRegression(),
        "Simple MLP": SimpleNeuralNetwork(),
        "Deep MLP": DeepMLP(),
        "Simple CNN": SimpleCNN(),
        "Complex CNN": ComplexCNN()
    }
    
    criterion = nn.CrossEntropyLoss()
    results = {}

    print("\n> Commencing Training & Evaluation...")
    for name, model in models.items():
        print(f"\n--- {name} ---")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("    Training...")
        train_model(model, train_loader, criterion, optimizer, epochs=EPOCHS)
        print("    Evaluating...")
        acc = evaluate_model(model, test_loader)
        print(f"    Test Accuracy: {acc:.2f}%")
        results[name] = acc

    print("\n==============================================")
    print("   COMPARISON STUDY RESULTS")
    print("==============================================")
    for name, acc in results.items():
        print(f"{name.ljust(20)}: {acc:.2f}% accuracy")

    # Plotting
    names = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies, color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    plt.ylim(0, 100)
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Model Comparison on {DATASET} ({EPOCHS} Epochs)')
    
    # Add value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.savefig('comparison_study.png')
    print(f"\n[SUCCESS] Comparison plot saved as 'comparison_study.png'")

if __name__ == '__main__':
    main()
