
# coding: utf-8

# In[30]:

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier


# In[11]:

df_train = pd.read_csv(r'C:\Users\Atin\Desktop\LIME_Seminar_Paper\Dataset\Sentiment -Amazon,IMDB,Yelp\Amazon_dataset_short\training_set_35.csv')
df_train


# In[13]:

df_test = pd.read_csv(r'C:\Users\Atin\Desktop\LIME_Seminar_Paper\Dataset\Sentiment -Amazon,IMDB,Yelp\Amazon_dataset_short\testing_set_15.csv')
df_test


# In[20]:

df_test.Data


# In[34]:

y_train = df_train['Target']
print(type(y_train))
print(np.asarray(y_train))
y_train = np.asarray(y_train)
print(y_train)


# In[37]:

y_test = df_test['Target']
print(type(y_test))
print(np.asarray(y_test))
y_test = np.asarray(y_test)
print(y_test)


# In[21]:

vectorizer = TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(df_train.Data)
test_vectors = vectorizer.transform(df_test.Data)


# In[22]:

train_vectors


# In[23]:

test_vectors


# In[24]:

print(vectorizer.get_feature_names())


# In[32]:

rf = RandomForestClassifier(n_estimators=500)
rf


# In[35]:

rf.fit(train_vectors, y_train)


# In[36]:

pred = rf.predict(test_vectors)
pred


# In[39]:

sklearn.metrics.f1_score(y_test, pred, average='binary')


# In[40]:

from lime import lime_text_new
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)


# In[46]:

print(c.predict_proba([df_test.Data[0]]))
###for single instance above 
###print(c.predict_proba(df_test.Data))


# In[48]:

from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer()
class_names = ['negative','positive']


# In[53]:

idx = 5
classfier_fn = c.predict_proba
exp = explainer.explain_instance(df_test.Data[idx], c.predict_proba, num_features=6,num_samples=1000)
print('Document id: %d' % idx)
print('Probability(positive) =', c.predict_proba([df_test.Data[idx]])[0,1])
print('True class: %s' % class_names[df_test.Target[idx]])


# In[51]:

exp.as_list()


# In[52]:

df_test.Data[idx]


# In[54]:

#get_ipython().magic('matplotlib inline')
fig = exp.as_pyplot_figure()


# In[55]:

exp.show_in_notebook(text=True)


# In[57]:

exp.as_map()


# In[ ]:



