import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from feature_engineering import get_cnn_features

class EmotionDataset(Dataset):
    def __init__(self, csv_file, label_encoder=None):
        self.df = pd.read_csv(csv_file)
        
        target_col = 'label' 
        
        if label_encoder is None:
            self.le = LabelEncoder()
            self.labels = self.le.fit_transform(self.df[target_col])
        else:
            self.le = label_encoder
            self.labels = self.le.transform(self.df[target_col])
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]['audio_path']
        features = get_cnn_features(audio_path)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label


class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()


        self.features = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), 
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




def train():

    train_dataset = EmotionDataset("data/train.csv")
    test_dataset = EmotionDataset("data/test.csv", label_encoder=train_dataset.le)
    

    joblib.dump(train_dataset.le, 'label_encoder.pkl')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(train_dataset.le.classes_) 
    model = EmotionCNN(num_classes=num_classes).to(device)
    
    print(f"Training on {num_classes} classes: {train_dataset.le.classes_}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_acc = 0.0
    model_path = "best_emotion_model.pth"

    for epoch in range(50):

        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        current_acc = 100 * correct / total
        scheduler.step(current_acc)
        

        if current_acc > best_acc:
            best_acc = current_acc

            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {current_acc:.2f}% | SAVED BEST MODEL")
        else:
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {current_acc:.2f}% | Best: {best_acc:.2f}%")


    torch.save(model.state_dict(), "final_emotion_model.pth")
    print(f"\nTraining Complete. Best Accuracy: {best_acc:.2f}%")
    print(f"Best model saved to: {model_path}")
    print("Label encoder saved to: label_encoder.pkl")

if __name__ == "__main__":
    train()
