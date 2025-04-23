# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.preprocessing import load_data, preprocess_data
# from src.modeling import train_model, evaluate_model
# from sklearn.model_selection import train_test_split

# import pandas as pd
# import logging
# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def main():
#     # Load and preprocess data
#     logging.info("Loading data...")
#     data = load_data()
#     logging.info("Preprocessing data...")
#     X, y = preprocess_data(data)

#     # Split data into training and testing sets
#     logging.info("Splitting data into train and test sets...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train the model
#     logging.info("Training the model...")
#     model = train_model(X_train, y_train)

#     # Evaluate the model
#     logging.info("Evaluating the model...")
#     evaluate_model(model, X_test, y_test)

# if __name__ == "__main__":
#     main()