import joblib   #для загрузки сохронения моделей

# Загружаем обученную модель и векторайзер
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Текст письма (можешь заменить на любое другое письмо   проверить: спам это или нет.)
sample_message = """From ilug-admin@linux.ie Tue Aug 13 10:29:44 2002 Return-Path: <ilug-admin@linux.ie> Delivered-To: yyyy@localhost.netnoteinc.com Received: from localhost (localhost [127.0.0.1]) by phobos.labs.netnoteinc.com (Postfix) with ESMTP id B92F744131 for <jm@localhost>; Tue, 13 Aug 2002 05:22:05 -0400 (EDT) Received: from phobos [127.0.0.1] by localhost with IMAP (fetchmail-5.9.0) for jm@localhost (single-drop); Tue, 13 Aug 2002 10:22:05 +0100 (IST) Received: from lugh.tuatha.org (root@lugh.tuatha.org [194.125.145.45]) by dogma.slashnull.org (8.11.6/8.11.6) with ESMTP id g7D8agb23313 for <jm-ilug@jmason.org>; Tue, 13 Aug 2002 09:36:42 +0100 Received: from lugh (root@localhost [127.0.0.1]) by lugh.tuatha.org (8.9.3/8.9.3) with ESMTP 
id JAA00835; Tue, 13 Aug 2002 09:35:58 +0100 
Received: from smtpout.xelector.com ([62.17.160.131]) by lugh.tuatha.org (8.9.3/8.9.3) with
ESMTP id JAA00802 for <ilug@linux.ie>; Tue, 13 Aug 2002 09:35:52 +0100 X-Authentication-Warning: 
lugh.tuatha.org: Host [62.17.160.131] claimed to be smtpout.xelector.com Received: from [172.18.80.234] 
(helo=xeljreilly) by smtpout.xelector.com with smtp (Exim 3.34 #2) id 17eWwo-0007CP-00; Tue, 
                                                     13 Aug 2002 09:22:38 +0100 Message-Id:"""

# Векторизуем сообщение
X = vectorizer.transform([sample_message])

# модель делает предсказание: спам (1) или не спам (0).
prediction = model.predict(X)[0]

# Выводим результат
print("Message (первые 300 символов):")
print(sample_message[:300] + "...")
print("\nPrediction:", "🚫 SPAM" if prediction == 1 else "✅ NOT SPAM")
