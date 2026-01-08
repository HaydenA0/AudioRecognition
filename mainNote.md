Objectif global : Concevoir un système de reconnaissance du locuteur à partir
du timbre vocal. Le système doit reconnaître qui parle, indépendamment du
contenu linguistique

Objectif spécifique : Une entreprise souhaite intégrer une authentification
vocale pour sécuriser l’accès à un service numérique. Votre mission est de
développer un prototype ML capable d’identifier un locuteur à partir d’un court
enregistrement audio.


la variable d’entrée du modèle : 
Caractéristiques clés de l'entrée audio
la variable cible :
identificateur du personne
le type de problème ML :
Classification supervisée multiclasse

Expliquer pourquoi la reconnaissance du locuteur est différente de la reconnaissance de la parole (ASR) : 
L'ASR sert à identifier ce qui est dit, mais ce problème consiste à savoir qui parle !

Lister au moins 3 facteurs pouvant perturber l’identification du locuteur :
- Le bruit, la qualité de la transmission et le ton du locuteur : si le ton est hors de la plage d'entraînement (données d'entraînement), la prédiction sera faible.


Pourquoi les features doivent être agrégées dans le temps ?
Ces caractéristiques sont calculées par frame, chaque frame ayant donc un seul objet caractéristique.
Nous souhaitons attribuer à chaque fichier audio (plusieurs frames) une seule caractéristique (vecteur fixe)
agrégée dans le temps, ce qui est naturel puisque l'audio est une série chronologique, ce qui facilite la
capture des propriétés globales du signal plutôt que des propriétés locales.

Répartition :
Échantillons d'entraînement : 1648 (locuteurs : 24)
Échantillons Val :   533 (locuteurs : 8)
Échantillons de test :  522 (locuteurs : 8)


Avec n_mfcc = 13 and C = 1.0
Training Accuracy: 0.5941
Test Accuracy: 0.3717

Avec n_mfcc = 20 and C = 1.0
Training Accuracy: 0.6388
Test Accuracy: 0.4126

avec n_mfcc = 20 and C = 11.0
Training Accuracy: 0.9861
Test Accuracy: 0.4219

avec n_mfcc = 20 and C = 11.0 and gamma = scale
Training Accuracy: 0.9861
Test Accuracy: 0.7613
