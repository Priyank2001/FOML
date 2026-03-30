import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath="train.csv"):
    print("Loading data for EDA...")
    df = pd.read_csv(filepath)
    # Kaggle's train.csv has 'label' as the first column
    y = df['label']
    X = df.drop('label', axis=1)
    return X, y

def plot_class_distribution(y):
    """Plots a bar chart to see if any digits are underrepresented."""
    plt.figure(figsize=(10, 5))
    sns.countplot(x=y, palette="viridis")
    plt.title("Digit Class Distribution in Training Set")
    plt.xlabel("Digit Label")
    plt.ylabel("Frequency")
    plt.show()

def plot_random_digits(X, y, num_digits=10):
    """Plots a random sample of digits so we can see human variations."""
    plt.figure(figsize=(15, 3))
    
    # Pick random rows
    random_indices = np.random.randint(0, len(X), num_digits)
    
    for i, idx in enumerate(random_indices):
        plt.subplot(1, num_digits, i + 1)
        # Reshape the flat 784 array back into a 28x28 image grid
        image = X.iloc[idx].values.reshape(28, 28)
        
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {y.iloc[idx]}")
        plt.axis('off')
        
    plt.suptitle("Sample Handwritten Digits", fontsize=16)
    plt.show()

def plot_average_digits(X, y):
    """Calculates and plots the 'average' representation of each digit."""
    plt.figure(figsize=(15, 6))
    
    for digit in range(10):
        # Filter the dataset for a specific digit
        digit_data = X[y == digit]
        
        # Calculate the mathematical average of every pixel across all images of this digit
        avg_image = digit_data.mean(axis=0).values.reshape(28, 28)
        
        plt.subplot(2, 5, digit + 1)
        plt.imshow(avg_image, cmap='magma')
        plt.title(f"Average '{digit}'")
        plt.axis('off')
        
    plt.suptitle("The 'Average' Pixel Intensity per Digit", fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    # 1. Load the dataset
    X, y = load_data("train.csv")
    
    # 2. Run the visualizations
    print("Generating Class Distribution chart...")
    plot_class_distribution(y)
    
    print("Generating Sample Digits grid...")
    plot_random_digits(X, y)
    
    print("Generating Average Digits chart...")
    plot_average_digits(X, y)

if __name__ == "__main__":
    main()