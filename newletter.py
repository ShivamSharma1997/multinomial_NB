from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

cat = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
ng_train = fetch_20newsgroups(subset = 'train', categories = cat)
ng_test = fetch_20newsgroups(subset = 'test', categories = cat)

vec = TfidfVectorizer()
vectors = vec.fit_transform(ng_train.data)

vectors_test = vec.transform(ng_test.data)

clf = MultinomialNB(alpha = 0.1)
clf.fit(vectors, ng_train.target)

pred = clf.predict(vectors_test)