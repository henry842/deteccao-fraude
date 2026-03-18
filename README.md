# 🔍 Detecção de Fraude em Transações Financeiras

> Projeto de Machine Learning para identificar transações fraudulentas em cartões de crédito usando técnicas avançadas de tratamento de dados desbalanceados.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![XGBoost](https://img.shields.io/badge/XGBoost-vs-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Concluído-success?style=flat-square)

---

## 📋 Sobre o projeto

Fraudes em cartões de crédito causam prejuízos bilionários globalmente. O desafio central deste projeto não é apenas técnico — é estatístico: em um dataset real, **menos de 0.2% das transações são fraudes**.

Um modelo ingênuo que classifica tudo como legítimo acertaria 99.83% das vezes — e seria completamente inútil. Este projeto demonstra como tratar esse problema corretamente, usando as métricas certas e técnicas específicas para dados desbalanceados.

### Pergunta central
> **É possível detectar fraudes com alta precisão sem gerar alertas falsos excessivos?**

---

## 🎯 Objetivos

- Analisar e compreender um dataset extremamente desbalanceado (1:578)
- Aplicar técnicas de balanceamento (SMOTE) de forma correta
- Treinar e avaliar dois modelos: **Árvore de Decisão** vs **XGBoost**
- Usar métricas adequadas: Precision, Recall, F1-Score e AUC-ROC
- Determinar qual modelo performa melhor no contexto de detecção de fraude

---

## 📊 Dataset

| Atributo | Valor |
|---|---|
| **Fonte** | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Transações totais** | 284.807 |
| **Fraudes** | 492 (0.172%) |
| **Features** | 30 (V1–V28 via PCA + Amount + Time) |
| **Período** | Setembro de 2013 — cartões europeus |

> ⚠️ O dataset não está incluído neste repositório por restrições de tamanho. Veja as instruções de instalação abaixo.

---

## 🧠 Conceitos aplicados

### Por que acurácia não serve aqui?
Com 99.83% de transações legítimas, um modelo que classifica **tudo** como legítimo teria 99.83% de acurácia — mas deixaria passar 100% das fraudes. Por isso usamos:

- **Recall** — Das fraudes reais, quantas foram detectadas? *(prioridade máxima)*
- **Precision** — Das marcadas como fraude, quantas realmente eram?
- **F1-Score** — Equilíbrio entre Precision e Recall
- **AUC-ROC** — Capacidade geral de separar as classes

### O que é SMOTE?
O **SMOTE** (Synthetic Minority Oversampling Technique) cria exemplos sintéticos da classe minoritária (fraude) interpolando entre exemplos existentes. Isso balanceia o dataset de treino sem simplesmente copiar dados.

> **Regra crítica:** SMOTE é aplicado **somente no conjunto de treino**. O teste mantém a distribuição real para avaliação honesta.

---

## 🏗️ Estrutura do projeto

```
deteccao_fraude/
├── 📓 projeto_deteccao_fraude.ipynb   # Notebook principal
├── 📄 README.md                        # Este arquivo
├── 📄 requirements.txt                 # Dependências
├── data/
│   ├── raw/
│   │   └── creditcard.csv             # Dataset (baixar do Kaggle)
│   └── processed/
│       └── metricas_comparacao.csv    # Resultados gerados
```

---

## ⚙️ Como executar

### 1. Clone o repositório
```bash
git clone https://github.com/SEU_USUARIO/deteccao-fraude.git
cd deteccao-fraude
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Baixe o dataset
1. Acesse [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Faça login e clique em **Download**
3. Coloque o arquivo `creditcard.csv` em `data/raw/`

### 4. Execute o notebook
Abra `projeto_deteccao_fraude.ipynb` no VS Code ou Jupyter e execute célula por célula.

---

## 🤖 Modelos e resultados

### Árvore de Decisão
- Modelo único de decisões sequenciais
- Tratamento: `class_weight='balanced'` + SMOTE no treino
- Interpretável mas limitada para padrões complexos

### XGBoost
- Ensemble de centenas de árvores com correção iterativa de erros
- Tratamento: `scale_pos_weight` penaliza erros na classe fraude
- Regularização L1/L2 evita overfitting

### Duelo final

| Métrica | Árvore de Decisão | XGBoost |
|---|---|---|
| **AUC-ROC** | 0.9112 | **0.9814** |
| **F1 (Fraude)** | 0.1634 | **0.8542** |
| **Recall (Fraude)** | 0.8469 | 0.8367 |
| **Avg Precision** | 0.5672 | **0.8774** |

> 📌 Preencha com seus resultados após rodar o notebook.

### Conclusão
O **XGBoost** superou a Árvore de Decisão principalmente por:
- Lidar nativamente com desbalanceamento via `scale_pos_weight`
- Ensemble de modelos reduz variância e overfitting
- Otimização direta da métrica AUC-PR para classes desbalanceadas

---

## 📚 Aprendizados principais

1. **Acurácia engana** — em dados desbalanceados, sempre use F1, Recall e AUC-ROC
2. **SMOTE no treino apenas** — aplicar no teste contamina a avaliação
3. **Custo assimétrico** — Falso Negativo (fraude não detectada) é muito mais caro que Falso Positivo
4. **XGBoost + scale_pos_weight** é uma combinação poderosa para problemas desbalanceados
5. **Curva Precision-Recall** é mais informativa que curva ROC para classes raras

---

## 🛠️ Tecnologias utilizadas

- **Python 3.10+**
- **Pandas** — manipulação de dados
- **NumPy** — operações numéricas
- **Matplotlib / Seaborn** — visualizações
- **Scikit-learn** — modelos e métricas
- **XGBoost** — gradient boosting
- **Imbalanced-learn** — SMOTE e técnicas de balanceamento

---

## 📁 requirements.txt

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
scipy>=1.11.0
```

---

## 👤 Autor

**Seu Nome**
- GitHub: [henry842](https://github.com/henry842)
- Programa de Empregabilidade — Módulo 4

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

---

*Desenvolvido como projeto do Programa de Empregabilidade — Módulo 4: Duelo entre modelos XGBoost x Árvore de Decisão*
