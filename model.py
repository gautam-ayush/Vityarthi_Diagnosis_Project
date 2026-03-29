import pandas as pd # panda library uses mathematics
from sklearn.tree import DecisionTreeClassifier # it helps to predict and program algorithm
from sklearn.model_selection import train_test_split
 # above are the libraries by which we can help our model to predict disease
 # for that time it will store data and tell disease then again it will get reset
class DiseaseModel:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = DecisionTreeClassifier()
        self.train_model() # model will get self trained

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
        # take inpur from the user and predict disease
        prediction = self.model.predict([symptoms])
        return prediction[0]

    def get_accuracy(self):
        return self.accuracy # retirn model accuracy
    # this is for training the model from which our model will work amd run according to that