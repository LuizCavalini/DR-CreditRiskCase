# Case Técnico — Cientista de Dados Júnior

## Apresentação

Este repositório contém o case técnico para o processo seletivo de **Cientista de Dados Júnior** na **Datarisk**, uma consultoria especializada em soluções baseadas em dados e inteligência aplicada ao mercado de crédito.
O desafio proposto envolve um projeto de **risco de crédito**, tema central de grande parte dos projetos que realizamos com nossos clientes.

## O Problema

O objetivo é prever a **probabilidade de inadimplência** de cobranças (títulos a pagar) de clientes, com base em dados cadastrais, informações mensais (renda, número de funcionários) e histórico de pagamentos. Considera-se inadimplente o pagamento realizado com **atraso igual ou superior a 5 dias** em relação à data de vencimento, ou **não realizado** (data de pagamento nula). O modelo treinado sobre o histórico de pagamentos deve ser usado para estimar essa probabilidade nas cobranças recentes, ainda sem desfecho conhecido.

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
