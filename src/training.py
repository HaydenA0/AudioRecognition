import pandas as pd
import numpy as np
from tqdm import tqdm
from preprocessor import preprocess_audio , load_audio
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def load_data(data_path) :
    training_data = pd.read_csv(data_path + "/train.csv")
    test_data = pd.read_csv(data_path + "/test.csv")
    return training_data, test_data

def extract_X_y(df, n_mfcc=20):
    X = []
    y = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            audio_path = row['audio_path']
            speaker_label = row['speaker_id']
            audio, sr = load_audio(audio_path)
            y_processed, sr_processed = preprocess_audio(audio, sr, show=False)
            from feature_engineering import extract_features, aggregate_features
            feat_matrix = extract_features(y_processed, sr_processed, n_mfcc=n_mfcc)
            vector = aggregate_features(feat_matrix)
            X.append(vector)
            y.append(speaker_label)
        except Exception as e:
            print(f"Error processing {e}")
            continue

    return np.array(X), np.array(y)


def train_and_evaluate_svm(X_train, y_train, X_test, y_test, verbose =False):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_test)
    model = SVC(kernel='rbf', C=11.0, gamma='scale', probability=True)

    print("Training...")
    model.fit(X_train_scaled, y_train)
    train_preds = model.predict(X_train_scaled)
    val_preds = model.predict(X_val_scaled)
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_test, val_preds)
    if verbose :
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {val_acc:.4f}")
    return model


def plot_confusion_matrix(y_true, y_pred, labels):
    fig, ax = plt.subplots(figsize=(10, 10))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title("Matrice de Confusion des Locuteurs")
    plt.show()






if __name__ == "__main__" :
    # PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROJECT_ROOT = "/home/anasr/dev/playground/ReconnaissanceAutomatiqueDuLocuteur"
    DATASET_PATH = PROJECT_ROOT + "/data"
    training_data, test_data = load_data(DATASET_PATH)
    X_train, labels_train = extract_X_y(training_data)
    X_test, labels_test = extract_X_y(test_data)

    model = train_and_evaluate_svm(X_train, labels_train, X_test, labels_test, verbose=True)
    val_preds = model.predict(X_test)
    plot_confusion_matrix(labels_test, val_preds, model.classes_)
    


