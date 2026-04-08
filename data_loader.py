import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_data_loaders(dataset_name='MNIST', batch_size=64):
    """
    Downloads the selected dataset (MNIST, FashionMNIST, KMNIST) and creates DataLoaders for PyTorch.
    """
    # Define a transform to normalize the data
    # Neural Networks perform better when pixel values are between -1 and 1 or 0 and 1
    # MNIST images are originally 0-255 grayscale values
    transform = transforms.Compose([
        transforms.ToTensor(), # converts images to PyTorch tensors and scales to [0.0, 1.0]
        transforms.Normalize((0.5,), (0.5,)) # mean 0.5, std 0.5 -> scales to [-1.0, 1.0]
    ])

    if dataset_name == 'MNIST':
        DatasetClass = datasets.MNIST
    elif dataset_name == 'FashionMNIST':
        DatasetClass = datasets.FashionMNIST
    elif dataset_name == 'KMNIST':
        DatasetClass = datasets.KMNIST
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"    [Sys] Processing primary image dataset: {dataset_name}...")
    train_set = DatasetClass(
        root='./data',       # Directory where the data will be saved
        train=True,          # Download the training split (60,000 images)
        download=True,       # Yes, download it!
        transform=transform  # Apply the transformations above
    )

    print(f"    [Sys] Processing testing dataset: {dataset_name}...")
    test_set = DatasetClass(
        root='./data',      
        train=False,         # Download the testing split (10,000 images)
        download=True,      
        transform=transform 
    )

    # DataLoaders help us iterate through the dataset in 'batches'
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def verify_data(train_loader):
    """
    Helper function to visualize the downloaded images to prove it works!
    """
    # Get a single batch of images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    print(f"Loaded a batch of shape: {images.shape}")
    print(f"Loaded a batch of labels shape: {labels.shape}")
    
    # Show the first 4 images in the batch
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        # un-normalize the image for viewing
        img = images[i].numpy().squeeze()
        img = img * 0.5 + 0.5
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis('off')
    
    plt.savefig('mnist_sample.png')
    print("Saved a sample image of the dataset to 'mnist_sample.png'")

if __name__ == '__main__':
    print("Testing DataLoader file directly...")
    for ds in ['MNIST', 'FashionMNIST', 'KMNIST']:
        train_loader, _ = get_data_loaders(dataset_name=ds)
        verify_data(train_loader)
        print(f"{ds} is ready successfully!")
