# Protocolo de revisão humana de rótulos

`scripts/build_fakewhatsapp_review_sample.py` produz uma amostra estratificada e cega: o arquivo não revela o rótulo original. Ele não cria nem altera rótulos.

Para cada item, uma pessoa revisora deve registrar uma URL de evidência verificável, decidir `verdadeiro`, `falso` ou deixar em branco quando não houver base suficiente, e explicar brevemente a decisão. Não use uma LLM para decidir o rótulo. Não trate a opinião do revisor como confirmação factual sem uma fonte externa.

Após a revisão, compare o `reviewer_label` com o rótulo original em um ambiente separado que contenha o CSV não cego. Divergências, itens inconclusivos e fontes ausentes precisam ser investigados antes de aceitar o dataset como avaliação externa. A fila e o resultado preenchido são dados de trabalho e permanecem ignorados pelo Git.
