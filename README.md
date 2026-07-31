# Case Técnico — Cientista de Dados Júnior

**Python:** 3.12.3

## Apresentação

Este repositório contém o case técnico para o processo seletivo de **Cientista de Dados Júnior** na **Datarisk**, uma consultoria especializada em soluções baseadas em dados e inteligência aplicada ao mercado de crédito.
O desafio proposto envolve um projeto de **risco de crédito**, tema central de grande parte dos projetos que realizamos com nossos clientes.

## Descrição do Problema

O objetivo é prever a **probabilidade de inadimplência** de cobranças (títulos a pagar) de clientes, com base em dados cadastrais, informações mensais (renda, número de funcionários) e histórico de pagamentos. Considera-se inadimplente o pagamento realizado com **atraso igual ou superior a 5 dias** em relação à data de vencimento, ou **não realizado** (data de pagamento nula). O modelo treinado sobre o histórico de pagamentos é usado para estimar essa probabilidade nas cobranças recentes, ainda sem desfecho conhecido (base de teste).

## Estrutura do Projeto

```
.
├── data/                                # bases brutas (não versionadas em git — ver "Como Executar")
│   ├── base_cadastral.csv
│   ├── base_info.csv
│   ├── base_pagamentos_desenvolvimento.csv
│   └── base_pagamentos_teste.csv
├── notebooks/
│   ├── 01_eda.ipynb                     # análise exploratória das bases
│   ├── 02_feature_engineering.ipynb     # validação do pipeline de features (sem vazamento de dados)
│   ├── 03_modelagem.ipynb               # baseline, LightGBM, tuning com Optuna e avaliação
│   └── 04_submissao.ipynb               # treino final e geração da submissão
├── src/
│   ├── __init__.py
│   └── feature_engineering.py           # pipeline de feature engineering (build_features)
├── requirements.txt                     # dependências com versões fixas
├── submissao_case.csv                   # arquivo final de submissão (gerado pelo notebook 04)
└── README.md
```

## Como Executar

1. **Crie e ative um ambiente virtual** (testado com Python 3.12.3):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Instale as dependências** (versões fixas, para reprodutibilidade):
   ```bash
   pip install -r requirements.txt
   ```
3. **Garanta que as 4 bases estejam na pasta `data/`** na raiz do projeto (`base_cadastral.csv`, `base_info.csv`, `base_pagamentos_desenvolvimento.csv`, `base_pagamentos_teste.csv`, todas com `sep=";"`).
4. **Execute os notebooks em ordem**, a partir da pasta `notebooks/`:
   1. `01_eda.ipynb` — análise exploratória
   2. `02_feature_engineering.ipynb` — geração e validação das features (checagem de vazamento de dados)
   3. `03_modelagem.ipynb` — baseline (Regressão Logística), LightGBM, tuning com Optuna e avaliação comparativa
   4. `04_submissao.ipynb` — treino do modelo final em toda a base de desenvolvimento e geração de `submissao_case.csv`

   Cada notebook pode ser aberto no Jupyter (`jupyter lab` / `jupyter notebook`) ou executado via linha de comando, por exemplo:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
   ```
5. Ao final da execução do notebook `04`, o arquivo **`submissao_case.csv`** é gerado na raiz do projeto.

## Abordagem Metodológica

1. **Análise exploratória (`01_eda.ipynb`):** entendimento da estrutura das quatro bases, qualidade dos dados (nulos, tipos, duplicidades), construção da regra de negócio da inadimplência e primeiras associações entre variáveis cadastrais/financeiras e o alvo.
2. **Feature engineering (`src/feature_engineering.py`, validado em `02_feature_engineering.ipynb`):** construção de **23 features** — cadastrais, mensais (renda/funcionários), comportamentais (histórico de pagamento do cliente) e da transação corrente. O ponto central do pipeline é a **prevenção de vazamento de dados (data leakage)**: as features comportamentais de uma linha usam, via `pandas.merge_asof` com `allow_exact_matches=False`, apenas pagamentos de safras **estritamente anteriores** à safra da própria linha — validado tanto por comparação com cálculo manual (força bruta) quanto por checagem de correlação e de crescimento monotônico do histórico ao longo do tempo.
3. **Modelagem (`03_modelagem.ipynb`):** split **temporal** (não aleatório) entre treino e validação, para refletir o cenário real de uso do modelo. Comparação entre um baseline de **Regressão Logística** (com imputação e padronização) e **LightGBM** (que lida nativamente com os nulos), com tuning de hiperparâmetros via **Optuna** (30 tentativas, otimizando AUC-ROC na validação temporal).
4. **Achado sobre calibração (`03`/`04`):** o uso de `is_unbalance=True` no LightGBM (comum para lidar com desbalanceamento de classes) melhora o ranqueamento (AUC/KS), mas **infla artificialmente as probabilidades previstas** — verificado comparando a probabilidade média prevista com a taxa real observada. Como a entrega exige uma probabilidade genuína (`PROBABILIDADE_INADIMPLENCIA`), o **modelo final foi treinado sem essa reponderação**, trocando uma perda marginal de AUC/KS por uma calibração correta.
5. **Geração da submissão (`04_submissao.ipynb`):** modelo final treinado em 100% da base de desenvolvimento, com as features geradas para a base de teste usando todo o histórico de desenvolvimento disponível, seguido de validações automáticas do arquivo de saída.

## Resultados

Métricas do modelo final (LightGBM, hiperparâmetros do Optuna, sem reponderação de classes) na validação temporal:

| Métrica | Valor |
|---|---|
| AUC-ROC | ~0.946 |
| KS | ~0.779 |
| Probabilidade média prevista vs. taxa real observada | praticamente idênticas (diferença de calibração desprezível) |

Optamos deliberadamente por essa configuração calibrada em vez da configuração com `is_unbalance=True` (AUC ~0.950, KS ~0.791): a diferença de poder de ranqueamento é pequena (menos de 1 ponto percentual em ambas as métricas), enquanto a diferença de calibração é enorme — a versão não calibrada chegou a prever, em média, mais que o dobro da taxa real de inadimplência. Para uma coluna de saída explicitamente chamada de "probabilidade", calibração correta pesou mais do que o ganho marginal de ranqueamento.

## Possíveis Melhorias

- **Validação walk-forward** com múltiplos cortes temporais, em vez de um único split treino/validação, para medir a estabilidade do modelo ao longo de diferentes períodos.
- **Target encoding com regularização** para variáveis de alta cardinalidade (`DDD`, `CEP_2_DIG`), como alternativa à frequência simples usada atualmente.
- **Calibração adicional** (Platt scaling ou Isotonic Regression) caso o uso do modelo exija probabilidades ainda mais precisas para fins regulatórios ou de precificação.
- **Janelas de histórico mais curtas** (últimos 3/6/12 meses) além do histórico acumulado total, para capturar mudanças recentes de comportamento do cliente.
- **Análise de erro por segmento** (porte, segmento industrial) para identificar subgrupos com performance sistematicamente pior.

## Instruções

Todas as regras, orientações e detalhes sobre a execução do case estão no documento:
📄 `Case DS Júnior 2025.pdf`

> O documento também contém **anexos com o dicionário de dados** e uma **visão dos relacionamentos entre as bases**, fundamentais para guiar sua análise.
> **Leia com atenção antes de iniciar sua solução.**

## Bases de Dados

Os arquivos estão disponíveis na pasta `/data` e incluem:

- `base_cadastral.csv`: informações cadastrais dos clientes, como porte, segmento industrial, CEP, e-mail e data de cadastro.
- `base_info.csv`: dados atualizados mensalmente com informações como renda do mês anterior e número de funcionários.
- `base_pagamentos_desenvolvimento.csv`: histórico de transações anteriores dos clientes, incluindo data de vencimento, valor a pagar, taxa e data de pagamento (quando disponível).
- `base_pagamentos_teste.csv`: transações recentes para as quais você deverá prever a probabilidade de inadimplência.

## Submissão

Você pode manter sua solução em um repositório pessoal para fins de portfólio, mas **a submissão oficial deve ser feita por e-mail**, conforme descrito no PDF, para garantir a **anonimidade na avaliação**.

💻 A solução deve ser desenvolvida e entregue obrigatoriamente em Python.

⚠️ **Não inclua informações pessoais (nome, LinkedIn, GitHub, etc.) nos arquivos entregues.**

## Recomendação

Mais do que aplicar técnicas avançadas, queremos entender **como você pensa, estrutura sua solução e toma decisões com base no problema de negócio**.
Soluções bem organizadas, com raciocínio claro e decisões justificadas são sempre valorizadas.

Seja também **curioso e criativo**: explore os dados com atenção e construa sua solução como se fosse apresentar para um cliente — explicando o que foi feito, por que foi feito e como sua proposta pode ser útil na prática.

**Boa sorte no desafio!**
