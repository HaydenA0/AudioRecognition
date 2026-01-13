import os
import pandas as pd


EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


STATEMENTS = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door"
}

def prepare_emotion_data(dataset_path):
    data = []
    for actor_dir in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor_dir)
        
        if not os.path.isdir(actor_path):
            continue
            

        for filename in os.listdir(actor_path):
            if filename.endswith(".wav"):

                parts = filename.replace(".wav", "").split("-")
                
                if len(parts) == 7:
                    emotion_code = parts[2]
                    intensity_code = parts[3]
                    statement_code = parts[4]
                    actor_id = parts[6]
                    
                    audio_path = os.path.join(actor_path, filename)
                    
                    data.append({
                        "actor_id": actor_id,
                        "emotion": EMOTIONS.get(emotion_code, "unknown"),
                        "intensity": "normal" if intensity_code == "01" else "strong",
                        "text": STATEMENTS.get(statement_code, "unknown"),
                        "audio_path": os.path.abspath(audio_path)
                    })

    df = pd.DataFrame(data)
    

    test_actors = ['21', '22', '23', '24']
    
    train_df = df[~df['actor_id'].isin(test_actors)]
    test_df = df[df['actor_id'].isin(test_actors)]
    
    return train_df, test_df

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(PROJECT_ROOT, "data/EmotionData/")

    train, test = prepare_emotion_data(DATASET_PATH)

    train.to_csv(os.path.join(PROJECT_ROOT, "data/train.csv"), index=False)
    test.to_csv(os.path.join(PROJECT_ROOT, "data/test.csv"), index=False)
    
    print(f"Data preparation complete. Train size: {len(train)}, Test size: {len(test)}")
