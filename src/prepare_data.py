from feature_engineering import PROJECT_ROOT
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(dataset_path):
    data = []
    for speaker_id in os.listdir(dataset_path):
        speaker_path = os.path.join(dataset_path, speaker_id)
        if not os.path.isdir(speaker_path):
            continue
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue
            trans_file = f"{speaker_id}-{chapter_id}.trans.txt"
            trans_path = os.path.join(chapter_path, trans_file)
            if os.path.exists(trans_path):
                with open(trans_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        file_id = parts[0]
                        text = parts[1]
                        audio_path = os.path.join(chapter_path, f"{file_id}.flac")
                        
                        data.append({
                            "speaker_id": speaker_id,
                            "audio_path": os.path.abspath(audio_path),
                            "text": text
                        })
    
    df = pd.DataFrame(data)
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['speaker_id']
    )

    
    return train_df, test_df


if __name__ == "__main__" :
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = PROJECT_ROOT + "/data/LibriSpeech/dev-clean/"
    train, test = prepare_data(DATASET_PATH)

    train.to_csv(PROJECT_ROOT + "/data/train.csv", index=False)
    test.to_csv(PROJECT_ROOT + "/data/test.csv", index=False)

    
