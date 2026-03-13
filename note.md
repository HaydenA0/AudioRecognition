 Ω time python src/training.py
Extracting features (first time)...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2162/2162 [00:49<00:00, 43.94it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 541/541 [00:12<00:00, 43.00it/s]
Training...
Training Accuracy: 1.0000
Test Accuracy: 0.9519

real	1m4.923s
user	11m51.880s
sys	0m2.629s


After caching
 Ω time python src/training.py
Cache Hit
Training...
Training Accuracy: 1.0000
Test Accuracy: 0.9519

real	0m2.830s
user	0m4.862s
sys	0m0.080s


C = 10.0
After now working with emotions and with minimal changes to the model :
Training Accuracy: 1.0000
Test Accuracy: 0.6285


C = 1.0
Training Accuracy: 0.8889
Test Accuracy: 0.5174

From the confusion matrix we can see that the model :
Happy is being confused with Angry, Fearful, and Sad.
Neutral and Sad are often confused with Calm.

To tell the difference between "Angry" and "Sad," the model needs to know how
loud (Energy) the person is and how high/low their pitch (Fundamental
Frequency) is.
For emotion, it is standard to use MFCCs 40, 
I used centroid and bandwidth, but for emotion, Zero Crossing Rate (how
"hissing" or "breathiness" there is) and Chroma (harmonic content) are very
important.

C = 1.0 and with changes
Training Accuracy: 0.8889
Test Accuracy: 0.5174

C = 10.0 and with changes
Training Accuracy: 1.0000
Test Accuracy: 0.6285

We can see that nothing chnaged why ?
Leakage there were a leakage in the data so that is why we got 100 % on
the trainning process

Now this is the true Accuracy of the model after fixing the leakage
Training Accuracy: 0.9675
Test Accuracy: 0.4167


MFCCs (especially 40 of them) are very "speaker-heavy." We need to add Pitch
(F0) and Spectral Contrast, which are much better at capturing emotion across
different people.


Use this chat :
https://aistudio.google.com/prompts/1BEvj1g8BAOPoDkB-YmOum3dyKNSHidC5
