#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# 

# In[95]:


import os
import missingno
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import *
from scipy.stats import *
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[96]:


data_path = os.path.join('..', 'data')
reports_path = os.path.join('..', 'reports')

print(f'Data path: {os.path.exists(data_path)} \nReports path: {os.path.exists(reports_path)}')


# In[97]:


df_path = os.path.join(data_path, 'raw/' 'spam_dataset.csv')
df = None

try:
    df = pd.read_csv(df_path, sep=',')
    print('Dataset has been initialized.')
except Exception as e:
    print(f'Failed to initialize dataset:\n{e}')


# In[98]:


df.head()


# In[99]:


df.info()


# In[100]:


if df.isna().sum().values.any():
    print(df.isna().sum())
    missingno.matrix(df)
else:
    print('There are no empty frames.')


# In[124]:


count_spam = df.value_counts(df['is_spam']).sort_values()
spam_percentage = count_spam.values[0]/len(df) * 100
print(f'Total emails: {len(df)}')
print(f'SPAM emails: {count_spam.values[0]} ({int(spam_percentage)}%)')
print(f'SPAM emails: {count_spam.values[1]} ({int(100 - spam_percentage)}%)')

plt.figure(figsize=(6, 5))
sns.barplot(x=count_spam.index, y=count_spam.values, palette=['green', 'red'])
plt.title('Spam/No spam ratio')
plt.xticks([0, 1], ['Regular', 'Spam'])
plt.xlabel(None)
plt.ylabel('Number of messages')
plt.grid(axis='y')
plt.legend()
plt.show()
plt.savefig(f'{reports_path}/RATIO_spam_not_spam.png')


# In[102]:


from wordcloud import WordCloud


# In[103]:


regular_content = " ".join(df[df['is_spam'] == 0]['text'])
spam_content = " ".join(df[df['is_spam'] == 1]['text'])


# In[104]:


# Wordcloud
wc = WordCloud(width=800, height=800, background_color='white', colormap='rocket', max_words=200)
spam_wc = wc.generate(spam_content)

# Wordcloud plot (for SPAM messages)
plt.figure(figsize=(8, 8))
plt.imshow(spam_wc , interpolation='bilinear')
plt.title('SPAM: words-inicators')
plt.axis('off')
plt.show()
plt.savefig(f'{reports_path}/spam_words.png')


# In[105]:


# Wordcloud
wc = WordCloud(width=800, height=800, background_color='white', colormap='rainbow', max_words=200)
spam_wc = wc.generate(regular_content)

# Wordcloud plot (for REGULAR messages)
plt.figure(figsize=(8, 8))
plt.imshow(spam_wc , interpolation='bilinear')
plt.title('REGULAR: words-inicators')
plt.axis('off')
plt.show()


# In[106]:


cont_tab_norm = pd.crosstab(index=df['is_spam'], columns=df['service'])
cont_tab_norm.index = ['Regular', 'Spam']

print('Observed values (service to spam):')
cont_tab_norm


# In[125]:


plt.figure(figsize=(8, 5))
sns.barplot(cont_tab_norm)
plt.title('Spam messages in every service')
plt.xlabel('Service')
plt.ylabel('Number')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
plt.savefig(f'{reports_path}/spam_msgs_by_service.png')


# In[122]:


chi2, p_value, dof, expected = chi2_contingency(cont_tab_norm)

print(f"Chi2: {chi2:.2f}, p-value: {p:.4f}")
if p < 0.05:
    print('The relationship is statistically significant.')
else:
    print('The relationship is NOT statistically significant.')

print('\nExpected values (service to spam):')
pd.DataFrame(expected, index=cont_tab_norm.index, columns=cont_tab_norm.columns)


# In[126]:


diff = cont_tab_norm - expected

plt.figure(figsize=(8, 5))
sns.heatmap(diff, annot=True, cmap="rocket", center=0)
plt.title("Service influence to spam (the difference between observed & expected values)")
plt.show()
plt.savefig(f'{reports_path}/CONTINGENCY_spam_msgs_by_service.png')

