import os
import glob
import pickle

ham_path = 'c:\\Documents\\Spam Detection ML\\enron1\\ham\\*'
spam_path = 'c:\\Documents\\Spam Detection ML\\enron1\\spam\\*'

emails, labels = [], []
print('Reading Ham files...\n')
for ham_file in glob.glob(ham_path):
	with open(ham_file, 'r', encoding="ISO-8859-1") as infile:
		emails.append(infile.read())
		labels.append(0)
print('Completed reading ham Files!\n')

print('Reading Spam Files...\n')
for spam_file in glob.glob(spam_path):
	with open(spam_file, 'r', encoding="ISO-8859-1") as infile:
		emails.append(infile.read())
		labels.append(1)
print('Completed reading spam Files!\n')
print('Reading all files done!\n')

print('writing files in Spam Detection ML folder..\n')
print('writing emails...')
with open('c:\\Documents\\Spam Detection ML\\emails.pkl', 'wb') as emails_file:
	pickle.dump(emails, emails_file)

print('writing labels...')
with open('c:\\Documents\\Spam Detection ML\\labels.pkl', 'wb') as labels_file:
	pickle.dump(labels, labels_file)

print('Completed writing in Spam Detection File!')
