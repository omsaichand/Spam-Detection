import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


with open('c:\\Documents\\Spam Detection ML\\emails_cleaned.pkl', 'rb') as emails_cleaned_file:
	emails_cleaned = pickle.load(emails_cleaned_file)

with open('c:\\Documents\\Spam Detection ML\\labels.pkl', 'rb') as labels_file:
	labels = pickle.load(labels_file)

#print(len(emails_cleaned))
#print(len(labels))
s=0
while(s<5):
	for i in range(len(emails_cleaned)):
		r = np.random.randint(i, len(emails_cleaned))
		temp_email = emails_cleaned[r]
		emails_cleaned[r] = emails_cleaned[i]
		emails_cleaned[i] = temp_email
		
		temp_label = labels[r]
		labels[r] = labels[i]
		labels[i] = temp_label
	s+=1
	
print(len(emails_cleaned))
print(len(labels))	

cv = CountVectorizer(stop_words = "english", max_features = None, max_df = 0.5, min_df = 20)

X_train, X_test, y_train, y_test = train_test_split(emails_cleaned, labels, test_size = 0.33, random_state = 42)

X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)
#y_train = labels
from sklearn.svm import SVC

clf = MultinomialNB(alpha = 20.0, fit_prior = False)
clf_svc = SVC(probability=True)
clf_svc.fit(X_train_cv, y_train)
predictions_prob = clf_svc.predict_proba(X_test_cv)
report = classification_report(y_test, clf_svc.predict(X_test_cv))
#score = cross_val_score(SVC(), X_train, y_train)
print(report)
print(predictions_prob)
roc_score = roc_auc_score(y_test, predictions_prob[:,1])
print(roc_score)
