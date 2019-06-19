#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
np.random.seed(1)


# In[2]:


df_train = pd.read_csv(r'C:\Users\Atin\Desktop\LIME_Seminar_Paper\Dataset\Sentiment -Amazon,IMDB,Yelp\All_train_1827.csv',encoding = "ISO-8859-1")
df_train


# In[3]:


df_test = pd.read_csv(r'C:\Users\Atin\Desktop\LIME_Seminar_Paper\Dataset\Sentiment -Amazon,IMDB,Yelp\All_test_914.csv',encoding = "ISO-8859-1")
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


#Model 1
rf = RandomForestClassifier(n_estimators=500)
rf


# In[9]:


rf.fit(train_vectors, y_train)


# In[10]:


pred = rf.predict(test_vectors)
pred


# In[11]:


sklearn.metrics.f1_score(y_test, pred, average='binary')


# In[12]:


#Model 2
model_logreg = LogisticRegression()
model_logreg.fit(train_vectors, y_train)


# In[13]:


pred2 = model_logreg.predict(test_vectors)
pred2


# In[14]:


sklearn.metrics.f1_score(y_test, pred2, average='binary')


# ## Lime Text Explainer

# In[15]:


from lime.lime_text_new import LimeTextExplainer as LimeTextExplainer_new
from lime.lime_text import LimeTextExplainer as LimeTextExplainer_old


from sklearn.pipeline import make_pipeline
c1 = make_pipeline(vectorizer, rf)
c2 = make_pipeline(vectorizer, model_logreg)


# In[16]:


print("Random_Forest",c1.predict_proba([df_test.Data[0]]))
print("Logistic_Regresseion",c2.predict_proba([df_test.Data[0]]))

###for single instance above 
###print(c.predict_proba(df_test.Data))


# In[17]:


#from lime.lime_text import LimeTextExplainer
explainer_old = LimeTextExplainer_old()
explainer_new = LimeTextExplainer_new()

class_names = ['negative','positive']


# In[18]:


len(df_test.Data)


# In[19]:


idx = np.random.randint(918)


# In[20]:


idx = 16


# In[21]:


#For RF
exp_old_rf = explainer_old.explain_instance(df_test.Data[idx], c1.predict_proba, num_features=6,num_samples=600)
exp_new_rf = explainer_new.explain_instance(df_test.Data,test_vectors,df_test.Data[idx],idx, c1.predict_proba,c1, num_features=6,num_samples=600)
print('Document id: %d' % idx)
print('Probability(positive) =', c1.predict_proba([df_test.Data[idx]])[0,1])
print('True class: %s' % class_names[df_test.Target[idx]])


# In[ ]:


#For LR
exp_old_lr = explainer_old.explain_instance(df_test.Data[idx], c2.predict_proba, num_features=6,num_samples=600)
exp_new_lr = explainer_new.explain_instance(df_test.Data,test_vectors,df_test.Data[idx],idx, c2.predict_proba,c2, num_features=6,num_samples=600)
print('Document id: %d' % idx)
print('Probability(positive) =', c2.predict_proba([df_test.Data[idx]])[0,1])
print('True class: %s' % class_names[df_test.Target[idx]])


# In[ ]:


#For RF


# In[ ]:


exp_old_rf.as_list()


# In[ ]:


exp_new_rf.as_list()


# In[ ]:


exp_old_rf.show_in_notebook(text=True)


# In[ ]:


exp_new_rf.show_in_notebook(text=True)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp_old_rf.as_pyplot_figure()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp_new_rf.as_pyplot_figure()


# In[ ]:


#For LR


# In[ ]:


exp_old_lr.show_in_notebook(text=True)


# In[ ]:


exp_new_lr.show_in_notebook(text=True)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp_old_lr.as_pyplot_figure()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = exp_new_lr.as_pyplot_figure()


# In[ ]:


exp_old.as_map()


# In[ ]:


exp_new.as_map()


# ### Lets try similar instances for prediction

# In[ ]:


idx = np.random.randint(918)
print(type(idx))
idx = np.int(25)
print(type(idx))


# In[ ]:


#For RF
exp_old_rf = explainer_old.explain_instance(df_test.Data[idx], c1.predict_proba, num_features=6,num_samples=600)
exp_new_rf = explainer_new.explain_instance(df_test.Data,test_vectors,df_test.Data[idx],idx, c1.predict_proba,c1, num_features=6,num_samples=600)
print('Document id: %d' % idx)
print('Probability(positive) =', c1.predict_proba([df_test.Data[idx]])[0,1])
print('True class: %s' % class_names[df_test.Target[idx]])


# In[ ]:


exp_old_rf.show_in_notebook(text=True)


# In[ ]:


exp_new_rf.show_in_notebook(text=True)


# In[ ]:


idx = 26
#For RF
exp_old_rf = explainer_old.explain_instance(df_test.Data[idx], c1.predict_proba, num_features=6,num_samples=600)
exp_new_rf = explainer_new.explain_instance(df_test.Data,test_vectors,df_test.Data[idx],idx, c1.predict_proba,c1, num_features=6,num_samples=600)
print('Document id: %d' % idx)
print('Probability(positive) =', c1.predict_proba([df_test.Data[idx]])[0,1])
print('True class: %s' % class_names[df_test.Target[idx]])


# In[ ]:


exp_old_rf.show_in_notebook(text=True)


# In[ ]:


exp_new_rf.show_in_notebook(text=True)


# In[ ]:





# In[ ]:




