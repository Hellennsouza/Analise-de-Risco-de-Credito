import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Simulação de Dados de Crédito com Variáveis Financeiras
np.random.seed(42)
n_records = 1500

data = pd.DataFrame({
    'idade': np.random.randint(21, 65, n_records),
    'renda_mensal': np.random.normal(5500, 2000, n_records).round(2),
    'score_credito': np.random.randint(300, 850, n_records),
    'taxa_utilizacao_limite': np.random.uniform(0.1, 1.0, n_records).round(2),
    'dias_atraso_historico': np.random.poisson(2, n_records),
})

# Regra de negócio sintética para inadimplência (1 = Inadimplente, 0 = Em dia)
prob_default = (
    (data['score_credito'] < 500) * 0.4 +
    (data['taxa_utilizacao_limite'] > 0.8) * 0.3 +
    (data['dias_atraso_historico'] > 3) * 0.3
)
data['inadimplente'] = (np.random.rand(n_records) < prob_default).astype(int)

# 2. Divisão Treino / Teste
X = data.drop(columns=['inadimplente'])
y = data['inadimplente']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Treinamento de Modelo Baseline (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Avaliação
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# 5. Importância das Variáveis para o Negócio
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== Principais Variáveis de Risco ===")
print(feature_importance)
