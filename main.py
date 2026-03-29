from model import DiseaseModel #importing model 
import os
# impoting modules from python in-built functions to help train and predict our model

def main(): #creating define function
    print(" Welcome to the AI Disease Diagnosis System!")
    print("Please answer the following questions in 1 and 0 \n")

    # Getting dataset path from which folder you will get access
    current_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(current_dir, "dataset.csv")

    # Loading and training model
    model = DiseaseModel(dataset_path)

    print(f"\n Model Accuracy: {model.get_accuracy() * 100:.2f}%\n")

    # Collect user inputs
    try: #input details of symptoms in form of 0 and 1
        fever =           int(input("Do you have Fever?"))
        cough =     int(input("Do you have Cough? "))
        headache   =  int(input("Do you have Headache? "))
        fatigue   =   int(input("Do you feel Fatigue? "))
        nausea =    int(input("Do you have Nausea? "))
    except ValueError: # enttering details if you have those symptoms or not
        print("\n Invalid input! Please enter only 0 or 1.")
        return

    # Store symptoms
    symptoms = [fever, cough, headache, fatigue, nausea] # these are the common symptoms for the any type of disease

    # Predict disease
    prediction = model.predict(symptoms)

    print("\n Based on your symptoms...")
    print(f" Predicted Disease: {prediction}")
    print("\n Note: This is a basic prediction. kindly consult a doctor for proper diagnosis.")

if __name__ == "__main__":
    main()
    # this is the model for identifying the symptoms and predict about the disease please kindly consult to a doctor