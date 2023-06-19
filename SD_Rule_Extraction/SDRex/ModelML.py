import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def prepare_model(sim_data, label_name, t_size):
    features = sim_data
    features = pd.get_dummies(features)
    labels = np.array(features[label_name])
    features = features.drop(label_name, axis=1)
    features = np.array(features)
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=t_size, random_state=42)
    return train_features, test_features, train_labels, test_labels


def train_model(train_features, train_labels, estimators=100):
    rf = RandomForestRegressor(n_estimators=estimators, random_state=42)
    rf.fit(train_features, train_labels)
    return rf


def predict_model(trained_model, test_features, test_labels):
    predictions = trained_model.predict(test_features)
    errors = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Accuracy: ', round(accuracy, 2), '%.')
    return predictions


def get_importance(model, dataframe):
    feats = {}  # a dict to hold feature_name: feature_importance
    for feature, importance in zip(dataframe.columns, model.feature_importances_):
        feats[feature] = importance  # add the name/value pair
    importance = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    return importance


def visualize_importance(importance):
    importance.sort_values(by='Gini-importance').plot(kind='bar', rot=45)
    plt.legend()
    plt.show()
