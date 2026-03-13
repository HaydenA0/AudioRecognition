import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_kaggle_data(dataset_path):
    data = []
    

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            

        label = folder_name.split('_')[0]
        
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                audio_path = os.path.join(folder_path, filename)
                
                data.append({
                    "label": label,
                    "audio_path": os.path.abspath(audio_path)
                })

    df = pd.DataFrame(data)
    


    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['label']
    )
    
    return train_df, test_df

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = PROJECT_ROOT + "/data/Audio_Dataset"
    train, test = prepare_kaggle_data(DATASET_PATH)
    
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    print(f"Prepared {len(train)} training and {len(test)} test samples.")
