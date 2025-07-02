#!/usr/bin/env python
# coding: utf-8

# # Model Training

# In[65]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
get_ipython().run_line_magic('matplotlib', 'inline')


# In[66]:


df_path = os.path.join('..', 'data', 'raw', 'spam_dataset.csv')
df = None

try:
    df = pd.read_csv(df_path, sep=',')
    print('Dataset has been initialized.')
except Exception as e:
    print(f'Failed to initialize dataset:\n{e}')


# In[67]:


df.head()


# In[68]:


df['combined'] = df['title'] + ' ' + df['text'] + ' ' + df['service']


# In[69]:


X = df['combined']
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


# In[70]:


vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# In[71]:


model = LogisticRegression()
model.fit(X_train_vec, y_train)


# In[72]:


features = vectorizer.get_feature_names_out()
coefs = model.coef_[0]

top_positive = np.argsort(coefs)[-10:]
top_negative = np.argsort(coefs)[:10]


print("SPAM words:")
for i in reversed(top_positive):
    print(f"{features[i]} ({coefs[i]:.2f})")

print('______')

print("\nRegular words:")
for i in top_negative:
    print(f"{features[i]} ({coefs[i]:.2f})")


# In[73]:


y_pred = model.predict(X_test_vec)


# In[74]:


print(classification_report(y_test, y_pred))


# In[75]:


cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Spam', 'Spam'],
            yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[79]:


import joblib


# In[82]:


models_path = os.path.join('..', 'src', 'models')

try:
    joblib.dump(model, f'{models_path}/spam_model.pkl')
    joblib.dump(vectorizer, f'{models_path}/vectorizer.pkl')
    print('SPAM MODEL has been successfully saved.')
except Exception as e:
    print(f'Failed to save the model:\n{e}')


# In[ ]:




