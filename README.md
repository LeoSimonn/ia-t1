# ia-t1

## Descrição

Este projeto implementa um sistema de IA para prever o resultado de jogos de Jogo da Velha (Tic-Tac-Toe) usando diferentes modelos de machine learning. Inclui:
- Pipeline de ciência de dados para treinar e comparar modelos.
- Interface gráfica em PyQt5 para jogar contra a IA e ver as previsões do modelo.

## Estrutura do Projeto

```
.
├── setup.ipy              # Pipeline de ciência de dados: treino, avaliação e salvamento do modelo
├── setup.py               # Interface gráfica (PyQt5) para jogar Jogo da Velha com IA
├── meu_modelo_treinado.pkl# Modelo treinado salvo (gerado pelo setup.ipy)
├── tic-tac-toe.data       # Base de dados original
├── tic-tac-toe.names      # Descrição dos atributos da base
├── README.md              # Este arquivo
```

## Dependências

- Python 3.8+
- pandas
- numpy
- scikit-learn
- joblib
- PyQt5

Instale as dependências com:
```bash
pip install pandas numpy scikit-learn joblib PyQt5
```

## Como treinar e salvar o modelo

1. Abra e execute o arquivo `setup.ipy` (pode ser em Jupyter Notebook ou Jupyter Lab).
2. O script irá baixar a base, balancear, treinar e comparar modelos.
3. O melhor modelo será salvo como `meu_modelo_treinado.pkl`.

## Como rodar a interface gráfica (PyQt5)

1. Certifique-se de que o arquivo `meu_modelo_treinado.pkl` existe (veja etapa anterior).
2. Execute:
   ```bash
   python setup.py
   ```
3. Jogue contra a IA! A interface mostra o tabuleiro, a previsão da IA, e estatísticas de acertos/erros.

## Como contribuir

- Faça um fork do repositório.
- Crie uma branch para sua feature/correção.
- Envie um Pull Request com uma descrição clara das mudanças.
- Siga as boas práticas de Python (PEP8).
- Teste seu código antes de enviar.

## Possíveis melhorias
- Adicionar mais modelos ou técnicas de ensemble.
- Melhorar a interface gráfica (cores, responsividade, etc).
- Adicionar testes automatizados.
- Permitir jogar contra diferentes modelos.
- Exportar estatísticas de partidas.

## Créditos
- Base de dados: [UCI Machine Learning Repository - Tic-Tac-Toe Endgame Data Set](https://archive.ics.uci.edu/ml/datasets/tic-tac-toe+endgame)

---

Colabore, melhore e divirta-se!