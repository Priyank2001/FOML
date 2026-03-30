from mnist_classifier import MNISTClassifier

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def run_experiment_svm(kernel_type, c_value, train_file, test_file, output_file):
    """Helper function to run an SVM and save its specific predictions."""
    svm_model = SVC(kernel=kernel_type,C=c_value)
    classifier = MNISTClassifier(svm_model)
    
    # We use None for test_samples to ensure we get all 28,000 Kaggle rows.
    # We keep train_samples at 10000 for faster experimenting.
    classifier.load_datasets(
        train_filepath=train_file, 
        test_filepath=test_file, 
        train_samples=10000, 
        test_samples=None 
    )
    
    classifier.train()
    
    # Capture the predictions returned by evaluate()
    predictions = classifier.evaluate()
    
    # Save this specific experiment's results to its own file
    classifier.save_submission(predictions, filename=output_file)
    print("-" * 50)
    del classifier
    del svm_model

def main():
    train_csv = "train.csv"
    test_csv = "test.csv"

    print("EXPERIMENT 1: Linear Kernel")
    run_experiment_svm(
        kernel_type='linear', 
        c_value=1.0, 
        train_file=train_csv, 
        test_file=test_csv,
        output_file="submission_linear.csv" # Generates file #1
    )

    print("\nEXPERIMENT 2: RBF Kernel")
    run_experiment_svm(
        kernel_type='rbf', 
        c_value=1.0, 
        train_file=train_csv, 
        test_file=test_csv,
        output_file="submission_rbf_c1.csv" # Generates file #2
    )
    
    print("\nEXPERIMENT 3: RBF Kernel (High Penalty)")
    run_experiment_svm(
        kernel_type='rbf', 
        c_value=10.0, 
        train_file=train_csv, 
        test_file=test_csv,
        output_file="submission_rbf_c10.csv" # Generates file #3
    )

if __name__ == "__main__":
    main()
