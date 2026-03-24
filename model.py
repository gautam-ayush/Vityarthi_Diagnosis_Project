import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

class DiseaseModel:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = DecisionTreeClassifier()
        self.train_model()

    def train_model(self):
        # Load dataset
        data = pd.read_csv(self.dataset_path)

        # Features and target
        X = data.drop("disease", axis=1)
        y = data["disease"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Store accuracy
        self.accuracy = self.model.score(X_test, y_test)

    def predict(self, symptoms):
        prediction = self.model.predict([symptoms])
        return prediction[0]

    def get_accuracy(self):
        return self.accuracy