# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris(as_frame=True)
df = pd.concat([data.data, data.target.rename('species')], axis=1)

# Descriptive statistics
def display_statistics(df):
    """Displays descriptive statistics for the dataset."""
    stats = df.describe()
    skewness = df.skew()
    kurtosis = df.kurtosis()
    print("Descriptive Statistics:\n", stats)
    print("\nSkewness:\n", skewness)
    print("\nKurtosis:\n", kurtosis)
    return stats, skewness, kurtosis

# Plotting a categorical plot (Bar Chart) for species counts
def plot_species_count(df):
    """Generates a bar chart for the count of each species."""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='species', palette="viridis")
    plt.title('Count of Each Species in the Dataset')
    plt.xlabel('Species')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# Plotting a relational plot (Scatter Plot) for Sepal Length vs Petal Length
def plot_scatter(df):
    """Generates a scatter plot showing Sepal Length vs Petal Length."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette="deep")
    plt.title('Relationship between Sepal Length and Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.tight_layout()
    plt.show()

# Plotting a statistical plot (Correlation Heatmap) of the correlation matrix
def plot_correlation_heatmap(df):
    """Generates a heatmap of the correlation matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Iris Features')
    plt.tight_layout()
    plt.show()

# Main function to run all analyses and plots
def main():
    """Main function to execute statistical analysis and generate plots."""
    # Display statistics
    stats, skewness, kurtosis = display_statistics(df)
    
    # Generate plots
    plot_species_count(df)
    plot_scatter(df)
    plot_correlation_heatmap(df)

# Run the main function
if __name__ == "__main__":
    main()
