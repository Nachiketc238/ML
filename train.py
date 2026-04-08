import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import SimpleNeuralNetwork

def train_model():
    """
    This is the core loop where the model learns how to recognize digits!
    It's left partially incomplete for you to finish as your assignment.
    """
    
    # 1. Hyperparameters: How the model learns
    epochs = 5        # Number of times we loop through the entire dataset
    learning_rate = 0.001 
    
    # 2. Setup Data and Model
    print("Setting up data and model...")
    # Load our MNIST dataset using the script we created
    train_loader, test_loader = get_data_loaders(batch_size=64)
    
    # Initialize the neural network
    model = SimpleNeuralNetwork()
    
    # 3. Define the Loss Function and Optimizer
    # CrossEntropyLoss is the standard loss function for classification problems (0-9)
    criterion = nn.CrossEntropyLoss()
    
    # Adam is a popular optimization algorithm (Stochastic Gradient Descent is another option!)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Beginning Training...")
    # 4. The main training loop
    for epoch in range(epochs):
        # We need to track our loss to see if the model is learning
        total_loss_for_epoch = 0.0 
        
        # Iterate over the dataset in batches
        for batch_index, (images, labels) in enumerate(train_loader):
            
            # --- YOUR TURN: THE MAGIC HAPPENS HERE ---
            # 
            # Step 1: Zero the gradients (clear previous calculations)
            optimizer.zero_grad() # I'll do the first one for you!
            
            # Step 2: Forward pass (pass the 'images' into the 'model' to get predictions)
            # predictions = ...
            
            # Step 3: Calculate the Loss (compare your predictions against the true 'labels' using 'criterion')
            # loss = ...
            
            # Step 4: Backward pass (calculate the gradient of the loss with respect to the model parameters)
            # loss.backward()
            
            # Step 5: Optimize (take a step in the direction that minimizes loss)
            # optimizer.step()
            
            # (Remove these lines once you've implemented the steps above)
            pass 
            loss = torch.tensor(0.0)
            # ----------------------------------------
            
            # Track the loss over time
            total_loss_for_epoch += loss.item()
            
            # Print a status update every 500 batches
            if batch_index % 500 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Batch {batch_index}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        # Calculate the average loss for this entire epoch
        avg_loss = total_loss_for_epoch / len(train_loader)
        print(f"--- End of Epoch {epoch+1}. Average Loss: {avg_loss:.4f} ---")
        
    print("\nTraining completed! (Don't forget to evaluate on your test data!)")

if __name__ == '__main__':
    train_model()
