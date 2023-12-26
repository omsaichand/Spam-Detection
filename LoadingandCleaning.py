import pickle
import nltk
nltk.download('wordnet')
#nltk.download('na')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names

with open('c:\\Documents\\Spam Detection ML\\emails.pkl', 'rb') as emails_file:
	emails = pickle.load(emails_file)

with open('c:\\Documents\\Spam Detection ML\\labels.pkl', 'rb') as labels_file:
	labels = pickle.load(labels_file)

names = list(names.words())
for index, name in enumerate(names):
	names[index] = name.lower()

lemmatizer = WordNetLemmatizer()
emails_cleaned = []
for email in emails:
	email = email.lower()
	email_cleaned= ' '.join(lemmatizer.lemmatize(word) for word in email.split() if word.isalpha() and word not in names)
	emails_cleaned.append(email_cleaned)

with open('c:\\Documents\\Spam Detection ML\\emails_cleaned.pkl', 'wb') as emails_cleaned_file:
	pickle.dump(emails_cleaned, emails_cleaned_file)
