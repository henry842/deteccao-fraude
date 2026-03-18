# Projeto Final — Detecção de Fraude em Transações Financeiras

**Autor:** Henry Rhuan Souza Santos  
**Programa:** Programa de Empregabilidade — EBAC  
**Entrega:** Módulo 41 — Duelo entre modelos: XGBoost x Árvore de Decisão  
**Repositório:** https://github.com/henry842/deteccao-fraude

---

## 1. Coleta de Dados

### Problemática

Fraudes em cartões de crédito representam um dos maiores desafios do setor financeiro mundial. No Brasil, o prejuízo causado por fraudes em meios de pagamento supera bilhões de reais por ano. A detecção precoce de transações fraudulentas é fundamental para proteger consumidores e instituições financeiras.

O problema central deste projeto não é apenas técnico — é estatístico: em um dataset real, menos de 0.2% das transações são fraudes. Um modelo ingênuo que classifica tudo como legítimo acertaria 99.83% das vezes e seria completamente inútil. Isso demonstra por que a escolha das métricas corretas é tão importante quanto a escolha do modelo.

**Pergunta central:** É possível construir um modelo de machine learning que identifique transações fraudulentas com alta precisão sem gerar alertas falsos excessivos?

### Justificativa do uso de dados

A análise de dados é a única abordagem escalável para esse problema. Um banco com milhões de transações por dia não pode ter analistas humanos revisando cada uma. Um modelo treinado com padrões históricos consegue avaliar milhares de transações por segundo, em tempo real, identificando anomalias que seriam invisíveis a olho nu.

### Fonte de dados

| Atributo | Descrição |
|---|---|
| **Dataset** | Credit Card Fraud Detection |
| **Fonte** | Kaggle — mlg-ulb/creditcardfraud |
| **Acesso** | Público e gratuito |
| **Transações** | 284.807 |
| **Fraudes** | 492 (0.172%) |
| **Período** | Setembro de 2013 — cartões europeus |
| **Features** | 30 (V1–V28 via PCA anonimizado + Amount + Time) |

As features V1 a V28 são componentes resultantes de uma transformação PCA aplicada para preservar a privacidade dos titulares dos cartões. As únicas features não transformadas são `Amount` (valor da transação) e `Time` (segundos desde a primeira transação do dataset).

### Desafio do desbalanceamento

O dataset apresenta um desbalanceamento extremo: para cada fraude, existem 578 transações legítimas (ratio 1:578). Isso exige técnicas específicas de tratamento, pois modelos treinados em dados desbalanceados tendem a ignorar a classe minoritária.

---

## 2. Modelagem

### Análise Exploratória de Dados (EDA)

A EDA revelou os seguintes pontos relevantes:

**Distribuição das classes:**
- Legítimas: 284.315 transações (99.827%)
- Fraudes: 492 transações (0.173%)
- Esta distribuição confirma a necessidade de técnicas de balanceamento

**Padrão temporal:**
- Fraudes ocorrem distribuídas ao longo do dia, sem concentração clara em horários específicos
- Transações legítimas têm picos nos horários comerciais

**Valor das transações:**
- Fraudes tendem a ter valores medianos menores que as legítimas
- Isso indica que fraudadores frequentemente testam valores menores para não acionar alertas

**Features mais discriminantes:**
- Através de teste t de Student, as features V4, V11, V12, V14 e V17 mostraram maior separação entre as classes
- Essas features foram as mais importantes nos modelos construídos

### Pré-processamento

**Normalização:** As colunas `Amount` e `Time` foram normalizadas com `StandardScaler`, pois estavam em escalas muito diferentes das features PCA.

**Balanceamento com SMOTE:**  
O SMOTE (Synthetic Minority Oversampling Technique) foi aplicado para criar exemplos sintéticos da classe fraude, equilibrando as classes no conjunto de treino.

> **Regra crítica aplicada:** O SMOTE foi aplicado **somente no conjunto de treino**. O conjunto de teste manteve a distribuição real (0.17% fraudes) para garantir uma avaliação honesta do modelo.

**Divisão dos dados:**
- Treino: 80% com estratificação (mantém proporção de fraudes)
- Teste: 20% com distribuição real

### Métricas escolhidas

A acurácia foi descartada como métrica principal por ser enganosa em dados desbalanceados. As métricas escolhidas foram:

- **Recall (Fraude):** Das fraudes reais, quantas foram detectadas? *(prioridade máxima — fraude não detectada = prejuízo)*
- **Precision (Fraude):** Das marcadas como fraude, quantas realmente eram?
- **F1-Score:** Equilíbrio entre Precision e Recall
- **AUC-ROC:** Capacidade geral do modelo de separar as classes

### Modelos construídos

**Modelo 1 — Árvore de Decisão:**
- `max_depth=10`, `min_samples_split=20`, `class_weight='balanced'`
- Treinada com dados balanceados via SMOTE
- Modelo interpretável mas limitado para padrões complexos

**Modelo 2 — XGBoost:**
- `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`
- `scale_pos_weight=578` — penaliza erros na classe fraude proporcionalmente ao desbalanceamento
- Treinado com dados originais (sem SMOTE), usando o parâmetro nativo de balanceamento
- Ensemble de centenas de árvores com correção iterativa de erros

### Resultados do duelo

| Métrica | Árvore de Decisão | XGBoost | Vencedor |
|---|---|---|---|
| **AUC-ROC** | 0.9112 | **0.9814** | XGBoost |
| **Avg Precision** | 0.5672 | **0.8774** | XGBoost |
| **F1 (Fraude)** | 0.1634 | **0.8542** | XGBoost |
| **Recall (Fraude)** | **0.8469** | 0.8367 | Árvore |

---

## 3. Conclusões

### Análise dos resultados

O **XGBoost venceu o duelo** em praticamente todas as métricas relevantes, com uma vantagem expressiva no F1-Score (0.8542 vs 0.1634) e no AUC-ROC (0.9814 vs 0.9112).

O resultado mais revelador está no F1-Score: a Árvore de Decisão obteve apenas 0.16, o que significa que apesar de detectar 84.69% das fraudes reais (Recall alto), ela gerou tantos alarmes falsos que sua Precision foi muito baixa. Na prática isso seria desastroso — o banco bloquearia centenas de transações legítimas para cada fraude detectada, gerando enorme insatisfação nos clientes.

O XGBoost equilibrou os dois lados: detectou 83.67% das fraudes reais com muito menos alarmes falsos, resultando em F1 de 0.85 — um modelo que funciona na prática.

### Por que XGBoost superou a Árvore de Decisão

1. **Ensemble vs modelo único:** XGBoost combina centenas de árvores, cada uma corrigindo os erros da anterior. A Árvore de Decisão é um único modelo com capacidade limitada
2. **scale_pos_weight:** O XGBoost penalizou erros na classe fraude 578 vezes mais que na classe legítima, alinhando o aprendizado ao custo real do negócio
3. **Regularização:** L1/L2 evitou overfitting, tornando o modelo mais robusto em dados novos
4. **Otimização direta:** Métrica `aucpr` otimizada especificamente para dados desbalanceados

### Aprendizados principais

1. **Acurácia engana** — um modelo com 99.83% de acurácia pode ser completamente inútil
2. **SMOTE somente no treino** — aplicar no teste contamina a avaliação e gera resultados irreais
3. **Custo assimétrico importa** — Falso Negativo (fraude não detectada) é muito mais caro que Falso Positivo
4. **F1-Score é a métrica síntese** — equilibra Precision e Recall em uma só métrica
5. **XGBoost + scale_pos_weight** é uma das combinações mais poderosas para problemas desbalanceados

### Próximos passos sugeridos

- Otimizar o threshold de decisão com base no custo financeiro real de cada tipo de erro
- Testar ADASYN e Borderline-SMOTE como alternativas ao SMOTE
- Implementar SHAP values para explicabilidade — essencial em aplicações financeiras reguladas
- Adicionar Random Forest como terceiro competidor no duelo
- Monitorar drift do modelo em produção (padrões de fraude mudam com o tempo)

---

## Autorização LGPD

Eu, **Henry Rhuan Souza Santos**, autorizo a cessão do meu projeto em favor da Semantix, bem como a divulgação do meu nome como autor responsável pelo projeto, uma vez que será possível incluir esse trabalho em meu portfólio de trabalho. Nesse sentido, autorizo também a divulgação dos meus contatos para a Semantix, tão somente para uso interno com finalidade única de contato em decorrência da elaboração do projeto mencionado.

- **E-mail:** henryitamaraca@gmail.com  
- **Telefone:** 71992683093

---

*Projeto desenvolvido no Programa de Empregabilidade EBAC — Módulo 41: Duelo entre modelos XGBoost x Árvore de Decisão*
