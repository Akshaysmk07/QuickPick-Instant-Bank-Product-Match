
# 🏦 QuickPick: Instant Bank Product Match

### 🔹 Description  
Speed up your decision-making with instant, tailored banking product suggestions that fit your unique financial profile.

---

### 📌 Summary  
![Weighted Hybrid Recommendation System](https://user-images.githubusercontent.com/61654792/217188533-4cc867f2-3888-4b7c-8028-c2971be6bafe.png)

The goal of this project is to create a more effective recommendation system. This allows the bank to better meet the individual needs of all customers. To achieve this, the user-item matrix will be used containing the ID of consumers and the products they owned as of May 28, 2015. Then, recommendations in three different recommendation models will be calculated:  
- Popularity-based  
- Memory-based collaborative filtering  
- Model-based collaborative filtering  

These models are combined into a **weighted hybrid recommendation system**, and evaluated using average precision metrics.

---

### 🧰 Technologies  
- Python  
- Scikit-Learn  
- Pandas  
- Numpy  
- Streamlit  
- Recommender Systems  

---

### 📂 Data Source  
Dataset from Kaggle competition:  
🔗 [Santander Product Recommendation](https://www.kaggle.com/competitions/santander-product-recommendation/data)

---

### ⭐ Popularity-Based Recommendation  
Calculates the probability of a product being owned.

```python
# A few products and their probability.
{'ind_ahor_fin_ult1': 0.0001,
 'ind_cco_fin_ult1': 0.775,
 'ind_cder_fin_ult1': 0.0005,
 'ind_cno_fin_ult1': 0.1003,
 'ind_ctju_fin_ult1': 0.0121}
```

---

### 🤝 Memory-Based Collaborative Filtering  
Uses cosine similarity to find users with similar behaviors.

```python
# only the most similar users
while k < 20:
    for user in range(len(df)):
        if sim_min < cosine_sim[cos_id, user] < 0.99:
            user_sim_k[user] = cosine_sim[cos_id, user]
            k += 1

    sim_min -= 0.025

    if sim_min < 0.65:
        break
```

---

### 🧠 Model-Based Collaborative Filtering  
Applies ML models to predict product probabilities.

```python
def modelbased(user_id, df, model=DecisionTreeClassifier(max_depth=9)):
    mdbs = {}
    
    for c in df.columns:
        y_train = df[c].astype('int')
        x_train = df.drop([c], axis=1)
        model.fit(x_train, y_train)
        p_train = model.predict_proba(x_train[x_train.index == user_id])[:,1]
        
        mdbs[c] = p_train[0]
        
    return mdbs
```

---

### 🔀 Weighted Hybrid  
Combines outputs from all 3 models using weights and gives final ranked recommendation.

---

### 📊 Evaluation  
Uses average precision @k, ROC-AUC, log-loss, and accuracy to measure model performance.
