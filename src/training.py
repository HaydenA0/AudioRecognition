import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV



def load_data(data_path) :
    training_data = pd.read_csv(data_path + "/train.csv")
    test_data = pd.read_csv(data_path + "/test.csv")
    return training_data, test_data

def extract_X_y(df, target_column='emotion'):
    X = []
    y = []
    
    if len(df) == 0:
        print("Error: The input DataFrame is empty!")
        return np.array([]), np.array([])

    for index, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row.get('audio_path')
        label = row.get(target_column)
        
        if audio_path is None or label is None:
            print(f"Error: Row {index} is missing 'audio_path' or '{target_column}'")
            continue

        if not os.path.exists(audio_path):
            print(f"Error: File not found at {audio_path}")
            continue

        try:
            from feature_engineering import get_char_vector
            from preprocessor import load_audio
            
            audio, sr = load_audio(audio_path)
            vector = get_char_vector(audio, sr)
            
            X.append(vector)
            y.append(label)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    return np.array(X), np.array(y)


def train_and_evaluate_svm(X_train, y_train, X_test, y_test, verbose=True):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)), 
        ('svc', SVC(kernel='rbf', probability=True))
    ])
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': [0.001, 0.01, 'scale'],
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    
    train_acc = accuracy_score(y_train, best_model.predict(X_train))
    val_acc = accuracy_score(y_test, best_model.predict(X_test))

    if verbose:
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {val_acc:.4f}")

    return best_model

def plot_confusion_matrix(y_true, y_pred, labels):
    fig, ax = plt.subplots(figsize=(10, 10))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title("Matrice de Confusion des Locuteurs")
    plt.show()




def load_X_labels(
    training_data,
    test_data,
    X_train_path,
    labels_train_path,
    X_test_path,
    labels_test_path
) :
    if os.path.exists(X_train_path):
        print("Cache Hit")
        X_train = np.load(X_train_path, allow_pickle=True)
        labels_train = np.load(labels_train_path, allow_pickle=True)
        X_test = np.load(X_test_path, allow_pickle=True)
        labels_test = np.load(labels_test_path, allow_pickle=True)
    else:
        print("Cache Miss")
        X_train, labels_train = extract_X_y(training_data) 
        X_test, labels_test = extract_X_y(test_data)
        np.save(X_train_path, X_train)
        np.save(X_test_path, X_test)
        np.save(labels_train_path, labels_train)
        np.save(labels_test_path, labels_test)
    return X_train, labels_train, X_test, labels_test


    # PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__" :
    PROJECT_ROOT = "/home/anasr/dev/playground/ReconnaissanceAutomatiqueDuLocuteur"
    DATASET_PATH = PROJECT_ROOT + "/data"
    
    CACHE_PATH = PROJECT_ROOT + "/cached/emotion_" 
    
    X_train_path = CACHE_PATH + "X_train.npy"
    labels_train_path = CACHE_PATH + "labels_train.npy"
    X_test_path = CACHE_PATH + "X_test.npy"
    labels_test_path = CACHE_PATH + "labels_test.npy"

    if not os.path.exists(os.path.dirname(CACHE_PATH)):
        os.makedirs(os.path.dirname(CACHE_PATH))

    training_data, test_data = load_data(DATASET_PATH)

    TARGET = 'emotion' 

    if os.path.exists(X_train_path):
        print("Cache Hit")
        X_train = np.load(X_train_path, allow_pickle=True)
        labels_train = np.load(labels_train_path, allow_pickle=True)
        X_test = np.load(X_test_path, allow_pickle=True)
        labels_test = np.load(labels_test_path, allow_pickle=True)
    else:
        print("Cache Miss")
        X_train, labels_train = extract_X_y(training_data, target_column=TARGET) 
        X_test, labels_test = extract_X_y(test_data, target_column=TARGET)
        np.save(X_train_path, X_train)
        np.save(X_test_path, X_test)
        np.save(labels_train_path, labels_train)
        np.save(labels_test_path, labels_test)

    # training 
    clf = train_and_evaluate_svm(X_train, labels_train, X_test, labels_test, verbose=True)
    
    # evaluating
    labels_pred = clf.predict(X_test)
    
    plot_confusion_matrix(labels_test, labels_pred, clf.classes_)
