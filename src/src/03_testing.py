#!/usr/bin/env python
# coding: utf-8

# # Testing

# In[3]:


import joblib
import os


# In[5]:


model_path = os.path.join('..', 'src', 'models', 'spam_model.pkl')
vectorizer_path = os.path.join('..', 'src', 'models', 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

print('Модель и векторизатор успешно загружены.')


# In[6]:


def predict_email(title, text, service):
    combined = title + " " + text + " " + service
    vec = vectorizer.transform([combined])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][1]
    
    return pred, proba


# In[33]:


emails = [
    {
        "title": "Поздравляем! Вы выиграли iPhone",
        "text": "Вы были выбраны для получения нового iPhone 15! Заполните анкету до конца дня.",
        "service": "Mail.ru"
    },
    {
        "title": "Результаты собеседования",
        "text": "Здравствуйте! Мы рады сообщить, что готовы предложить вам должность финансового директора.",
        "service": "Gmail"
    },
    {
        "title": "Ваша карта заблокирована",
        "text": "Чтобы восстановить доступ, пройдите верификацию по ссылке. Это займёт максимум 30 секунд!",
        "service": "Gmail"
    },
    {
        "title": "Изменения в тарифе",
        "text": "Введены новые изменения в тарифе Билайн. Перейдите по ссылке для получения информации.",
        "service": "Mail.ru"
    }
]


# In[34]:


for email in emails:
    label, proba = predict_email(email['title'], email['text'], email['service'])
    label_str = '🚨СПАМ🚨' if label else '✅НЕ спам✅'
    print(f"📩 '{email['title']}' — {label_str} (вероятность спама: {int(proba * 100)}%)")


# In[ ]:




