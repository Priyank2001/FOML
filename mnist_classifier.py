import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class MNISTClassifier:
    def __init__(self, model):
        self.model = model
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _process_dataframe(self, df, sample_size):
        """Helper method to extract features/labels and scale them."""
        if df.shape[1] == 784:
            y = None  # Kaggle test set (no labels)
            X = df.values
        elif "label" in df.columns:
            y = df["label"].values
            X = df.drop("label", axis=1).values
        else:
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values

        # If sample_size is None, it grabs all rows. Otherwise, it subsets.
        X_subset = X[:sample_size] / 255.0
        y_subset = y[:sample_size] if y is not None else None

        return X_subset, y_subset

    def load_datasets(
        self, train_filepath, test_filepath, train_samples=None, test_samples=None
    ):
        """Loads data. Pass 'None' to sample sizes to load the entire CSV."""
        print(f"Loading training data...")
        df_train = pd.read_csv(train_filepath)
        self.X_train, self.y_train = self._process_dataframe(df_train, train_samples)

        print(f"Loading testing data...")
        df_test = pd.read_csv(test_filepath)
        self.X_test, self.y_test = self._process_dataframe(df_test, test_samples)

        print(
            f"Data ready: {len(self.X_train)} training and {len(self.X_test)} testing samples."
        )

    def train(self):
        if self.X_train is None:
            raise ValueError("Data not loaded.")

        print(f"\nTraining SVM with '{self.model.kernel}' kernel (C={self.model.C})...")
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    def evaluate(self):
        """Tests the model and returns the predictions."""
        if self.X_test is None:
            raise ValueError("Data not loaded.")

        print("Generating predictions...")
        y_pred = self.model.predict(self.X_test)

        if self.y_test is not None:
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"Accuracy: {accuracy:.2%}")
        else:
            print("Notice: No labels in test set. Bypassing accuracy check.")

        return y_pred

    def save_submission(self, predictions, filename="submission.csv"):
        """Formats the predictions and saves them for Kaggle."""
        print(f"Saving format for Kaggle to -> {filename}")

        # Kaggle requires ImageId to start at 1
        image_ids = range(1, len(predictions) + 1)

        # Create a Pandas DataFrame
        submission_df = pd.DataFrame({"ImageId": image_ids, "Label": predictions})

        # Save to CSV (index=False prevents Pandas from adding its own row numbers)
        submission_df.to_csv(filename, index=False)
        print("Save complete!")
