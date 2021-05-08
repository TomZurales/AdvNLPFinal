import xmlschema
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

article_schema = xmlschema.XMLSchema('data/article.xsd')
ground_truth_schema = xmlschema.XMLSchema('data/ground-truth.xsd')

articles_raw = article_schema.to_dict('data/byarticle/articles-training-byarticle-20181122.xml')['article']
ground_truths_raw = ground_truth_schema.to_dict('data/byarticle/ground-truth-training-byarticle-20181122.xml')['article']

articles = []
labels = []

for i in range(len(articles_raw)):
    article = articles_raw[i]
    labels.append(ground_truths_raw[i]['@hyperpartisan'])
    text = article['@title'].lower()
    for key in article.keys():
        if not key.startswith('@'):
            for t in article[key]:
                if str(t)[0] not in ['{', '[']:
                    text = text + ' ' + str(t).lower()
    articles.append(text)

labels = [i*1 for i in labels]

d = {'Article': articles, 'Label': labels}
df = pd.DataFrame(d)

X_train, X_test, y_train, y_test = train_test_split(df['Article'], df['Label'], random_state=1)

cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)

print('Accuracy score: ', accuracy_score(y_test, predictions))
print('Precision score: ', precision_score(y_test, predictions))
print('Recall score: ', recall_score(y_test, predictions))