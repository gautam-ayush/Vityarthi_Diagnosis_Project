from model import DiseaseModel
import os

def main():
    print("🧠 Starting Disease Diagnosis System...\n")

    # Get correct path of dataset.csv
    current_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(current_dir, "dataset.csv")

    # Load model with dataset
    model = DiseaseModel(dataset_path)

    print(f"Model Accuracy: {model.get_accuracy()*100:.2f}%")

    print("\nEnter symptoms (1 = Yes, 0 = No)\n")

    fever = int(input("Fever: "))
    cough = int(input("Cough: "))
    headache = int(input("Headache: "))
    fatigue = int(input("Fatigue: "))
    nausea = int(input("Nausea: "))

    symptoms = [fever, cough, headache, fatigue, nausea]

    prediction = model.predict(symptoms)

    print("\n🩺 Predicted Disease:", prediction)


if __name__ == "__main__":
    main()