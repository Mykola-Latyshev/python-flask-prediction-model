# Flask_with_model
Welcome to our Git Hub repository! Here, you will find a Python Flask local server that uses a machine learning model trained on XGBoost to predict diseases based on various features. There are three models included in the repository, each of which uses a different number of features: the first model uses all available features, the second uses 11 specific features, and the third predicts using only three features.

In addition, the repository contains a client file for use in Jupyter Notebook. By using this file, you can easily access the server and receive predictions from the model. The machine learning model is stored as a pickle file within the repository's folder.

The dataset used for training and testing the model was obtained from the website https://www.openml.org/d/35. The problem at hand was that of multi-class classification, where the number of classes is exactly six (class=6), each of which corresponds to a specific disease. The client outputs the three most probable diseases along with their corresponding probabilities.
