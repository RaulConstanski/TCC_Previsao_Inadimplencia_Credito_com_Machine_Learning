# 💳 Previsão de Inadimplência de Cartão de Crédito com Machine Learning

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

Projeto de conclusão de curso (TCC) focado no desenvolvimento e comparação de modelos preditivos para identificar o risco de inadimplência de clientes de cartão de crédito no mês subsequente. O objetivo prático é antecipar o risco financeiro para subsidiar ações preventivas de cobrança, negociação e otimização da Provisão de Devedores Duvidosos (PDD).

O Artigo completo pode ser visualizada dentro da pasta template neste projeto.

---

## Contexto e Cenário

* **Desafio de Negócio:** Toda instituição financeira lida com inadimplência, é um problema relacionado a essência do negócio, ela empresta e precisa receber esse valor emprestádo de volta. Aumento de inadimplência leva ao aumento de PDD e a prejuízos, especialmente em cartão de crédito, onde não há garantia atrelada a esse empréstimo.
* **A Base de Dados:** Análise sobre 30.000 registros e 25 variáveis (*Default of Credit Card Clients*).
* **Desbalanceamento das Classes:** A base apresenta 22,12% de clientes inadimplentes (Classe 1) contra 77,88% adimplentes (Classe 0).

### Grupos de Variáveis Analisadas
1. **Demográficas:** Idade, sexo, educação e estado civil.
2. **Financeiras e Históricas (Últimos 6 meses):** Limite de crédito concedido, evolução dos saldos de faturas, comportamento de status de pagamento (atrasos) e valores efetivamente pagos.

---

## Limpeza de Dados e Pré-processamento

O pipeline de estruturação de dados seguiu três etapas fundamentais para garantir a consistência estatística:

1. **Saneamento e Agrupamento:** Correção de registros não documentados nas variáveis de escolaridade e estado civil para categorias genéricas ("Outros"). Manutenção e validação dos códigos `-2` e `0` na série temporal de pagamentos após comprovação analítica de que representavam comportamento financeiro saudável (88% adimplentes).
2. **Data Splitting Estratificado:** Divisão rigorosa em **60% Treino**, **20% Validação** e **20% Teste**, preservando a proporção exata da classe alvo nas três partições (*stratify*).
3. **Transformações Codificadas:** Aplicação de `StandardScaler` (Z-score) para normalizar variáveis numéricas de faturamento e limites, `OneHotEncoder` para variáveis categóricas binárias e balanceamento artificial de treino via técnicas de peso de classe (`Class Weight`).

---

## Modelos Avaliados e Arquitetura

Foram testadas e comparadas diferentes abordagens algorítmicas para solucionar o problema:
* **Regressão Logística:** Utilizada como modelo baseline linear de alta interpretabilidade.
* **SVM (Kernel RBF):** Focado na maximização de margens de decisão e captura de fronteiras não lineares.
* **Random Forest & XGBoost:** Estruturas baseadas em árvores (*Bagging* e *Boosting* iterativo) para mitigação de overfitting e extração de padrões complexos.
* **Voting Classifier (Ensemble):** Modelo híbrido via *soft voting* unindo Regressão Logística, Random Forest e XGBoost.
* **LSTM (Deep Learning):** Rede neural recorrente configurada para capturar dependências e o comportamento temporal histórico das faturas.

---

## Análise de Resultados (Dados de Teste)

Dado o desbalanceamento da base, o **F1-Score (Classe 1)** foi eleito como a métrica norteadora do negócio para equilibrar o custo de falsos positivos e a perda por falsos negativos.

| Modelo | Acurácia | Precisão (Inad.) | Recall (Inad.) | F1-Score (Inad.) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Regressão Logística | 0,676 | 0,366 | **0,635** | 0,464 | 0,712 |
| **SVM (Melhor F1)** | 0,775 | 0,492 | 0,558 | **0,523** | 0,751 |
| Random Forest | 0,813 | **0,647** | 0,335 | 0,442 | 0,756 |
| XGBoost | 0,755 | 0,455 | 0,539 | 0,493 | 0,746 |
| Voting Classifier | 0,799 | 0,555 | 0,472 | 0,510 | 0,757 |
| LSTM (Maior AUC) | **0,818** | 0,648 | 0,386 | 0,484 | **0,770** |

### Conclusões Principais
* **O Modelo Campeão:** O algoritmo **SVM (Support Vector Machine)** apresentou a melhor aderência ao objetivo do negócio com um **F1-Score de 0,523**, garantindo o equilíbrio mais seguro entre a confiabilidade da predição e o volume de inadimplentes capturados.
* **Comportamento Temporal:** A análise exploratória (EDA) confirmou que atrasos recentes (`PAY_1` com correlação de 0,32) exercem o maior peso preditivo no risco iminente de default.
* **Trade-off de Redes Neurais:** O modelo LSTM alcançou a maior área sob a curva (ROC-AUC de 0,770) e ótima acurácia geral, porém penalizou severamente o *Recall* (0,386), deixando escapar muitos clientes de risco.

---

## Autores

* **Camille M. Pleger** - [E-mail](mailto:camille.pleger@pucpr.edu.br)
* **Raul C. Rodrigues** - [E-mail](mailto:raul.constanski@pucpr.edu.br)

Instituição: **Pontifícia Universidade Católica do Paraná (PUCPR)** - 2026.