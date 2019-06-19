#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
np.random.seed(1)


# In[2]:


df_train = pd.read_csv(r'C:\Users\Atin\Desktop\LIME_Seminar_Paper\Dataset\Sentiment -Amazon,IMDB,Yelp\Amazon_train_800.csv')
df_train


# In[3]:


df_test = pd.read_csv(r'C:\Users\Atin\Desktop\LIME_Seminar_Paper\Dataset\Sentiment -Amazon,IMDB,Yelp\Amazon_test_200.csv')
df_test


# In[4]:


df_test.Data


# In[5]:


y_train = df_train['Target']
print(type(y_train))
print(np.asarray(y_train))
y_train = np.asarray(y_train)
print(y_train)


# In[6]:


y_test = df_test['Target']
print(type(y_test))
print(np.asarray(y_test))
y_test = np.asarray(y_test)
print(y_test)


# In[7]:


vectorizer = TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(df_train.Data)
test_vectors = vectorizer.transform(df_test.Data)


# In[8]:


test_vectors


# In[9]:


from sklearn.neighbors import NearestNeighbors


# In[10]:


nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(test_vectors)


# In[11]:


distances, indices = nbrs.kneighbors(test_vectors)
print(distances)
print(indices)


# In[12]:


test_vectors[indices[0]].todense()


# In[13]:


distances[0]


# In[14]:


import scipy
a = scipy.sparse.csr_matrix([[1, 0, 0, 0], [0, 0, 10, 11], [0, 0, 0, 99]])


# In[15]:


a[2].todense()


# In[16]:


len(vectorizer.get_feature_names())


# In[17]:


distances


# In[ ]:





# In[ ]:





# In[18]:


rf = RandomForestClassifier(n_estimators=500)
rf


# In[19]:


rf.fit(train_vectors, y_train)


# In[20]:


pred = rf.predict(test_vectors)
pred


# In[21]:


pred_n = rf.predict(test_vectors[indices[0]])
pred_n


# In[22]:


pred_prob_n = rf.predict_proba(test_vectors[indices[0]])
pred_prob_n


# In[23]:


sklearn.metrics.f1_score(y_test, pred, average='binary')


# In[24]:


T =np.asarray(test_vectors[indices[0]].todense())


# In[25]:


type(T)


# In[26]:


doc = 0
feature_index = test_vectors[doc,:].nonzero()
tfidf_scores = zip(feature_index, [test_vectors[doc, x] for x in feature_index])


# In[27]:


feature_index


# In[28]:


vectorizer.get_feature_names()[1920]


# In[29]:


temp_text = vectorizer.transform(["Good , works fine."])


# In[30]:


temp_text[doc,:].nonzero()


# In[31]:


for i, v in enumerate(tfidf_scores):
    print(v)


# In[32]:





# In[33]:


vectorizer.get_feature_names()


# In[44]:


from lime.lime_text_new  import LimeTextExplainer as LimeTextExplainer_new
from lime.lime_text import LimeTextExplainer as LimeTextExplainer_old


from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)


# In[45]:


print(c.predict_proba([df_test.Data[0]]))
###for single instance above 
###print(c.predict_proba(df_test.Data))


# In[46]:


#from lime.lime_text import LimeTextExplainer
explainer_old = LimeTextExplainer_old()
explainer_new = LimeTextExplainer_new()

class_names = ['negative','positive']


# In[47]:


len(df_test.Data)


# In[49]:


idx = 17
exp_old = explainer_old.explain_instance(df_test.Data[idx], c.predict_proba, num_features=6,num_samples=100)
exp_new = explainer_new.explain_instance(df_test.Data,test_vectors,df_test.Data[idx],idx
                                         ,c.predict_proba,c , num_features=6,num_samples=100)
print('Document id: %d' % idx)
print('Probability(positive) =', c.predict_proba([df_test.Data[idx]])[0,1])
print('True class: %s' % class_names[df_test.Target[idx]])


# In[ ]:

exp_old.as_list()

exp_new.as_list()


# In[ ]:


c.predict_proba([df_test.Data[idx]])


# In[ ]:


df_test.Data[idx]


# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp_new.as_pyplot_figure()


# In[ ]:


exp_new.show_in_notebook(text=True)


# In[ ]:


exp_new.as_map()


# ### Lets try similar instances for prediction

# In[ ]:


idx = 18
#exp = explainer.explain_instance(df_test.Data[idx], c.predict_proba, num_features=6,num_samples=5000)
print('Document id: %d' % idx)
print('Probability(positive) =', c.predict_proba([df_test.Data[idx]])[0,1])
print('True class: %s' % class_names[df_test.Target[idx]])


# In[ ]:


exp.as_list()


# In[ ]:


exp.show_in_notebook(text=True)


# In[ ]:


idx = 19
exp = explainer.explain_instance(df_test.Data[idx], c.predict_proba, num_features=2,num_samples=5000)
print('Document id: %d' % idx)
print('Probability(positive) =', c.predict_proba([df_test.Data[idx]])[0,1])
print('True class: %s' % class_names[df_test.Target[idx]])


# In[ ]:


exp.as_list()


# In[ ]:


exp.show_in_notebook(text=True)


# In[ ]:




