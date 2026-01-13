import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from feature_engineering import get_cnn_features


class EmotionDataset(Dataset):
    def __init__(self, csv_file, label_encoder=None):
        self.df = pd.read_csv(csv_file)
        if label_encoder is None:
            self.le = LabelEncoder()
            self.labels = self.le.fit_transform(self.df['emotion'])
        else:
            self.le = label_encoder
            self.labels = self.le.transform(self.df['emotion'])
            
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

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(0.3)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train():
    train_dataset = EmotionDataset("data/train.csv")
    test_dataset = EmotionDataset("data/test.csv", label_encoder=train_dataset.le)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
        
        print(f"Epoch {epoch+1}/50, Loss: {total_loss/len(train_loader):.4f}, Test Acc: {100 * correct / total:.2f}%")
if __name__ == "__main__":
    train()
