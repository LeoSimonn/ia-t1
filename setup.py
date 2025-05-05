import sys
import numpy as np
import random
import joblib
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Carregue seu modelo treinado aqui
modelo = joblib.load('meu_modelo_treinado.pkl')
columns = [f'pos{i}' for i in range(1, 10)]

class JogoVelha(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Jogo da Velha IA')
        self.estado = np.zeros((3, 3), dtype=int)
        self.acertos = 0
        self.erros = 0
        self.total = 0
        self.fim = False
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        # Tabuleiro
        self.grid_layout = QGridLayout()
        self.botoes = [[None for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                btn = QPushButton('')
                btn.setFixedSize(70, 70)
                btn.setFont(QFont('Arial', 24))
                btn.clicked.connect(lambda _, x=i, y=j: self.jogada_usuario(x, y))
                self.grid_layout.addWidget(btn, i, j)
                self.botoes[i][j] = btn
        main_layout.addLayout(self.grid_layout)
        # Status
        self.status_label = QLabel('Ainda em jogo')
        self.status_label.setFont(QFont('Arial', 14, QFont.Bold))
        main_layout.addWidget(self.status_label, alignment=Qt.AlignCenter)
        # Previsão da IA
        ia_layout = QHBoxLayout()
        ia_label_static = QLabel('Previsão da IA:')
        ia_label_static.setFont(QFont('Arial', 12))
        self.ia_label = QLabel('Ainda em jogo')
        self.ia_label.setFont(QFont('Arial', 12))
        self.ia_label.setStyleSheet('color: blue;')
        ia_layout.addWidget(ia_label_static)
        ia_layout.addWidget(self.ia_label)
        main_layout.addLayout(ia_layout)
        # Estatísticas
        stats_layout = QHBoxLayout()
        self.acertos_label = QLabel('Acertos: 0')
        self.erros_label = QLabel('Erros: 0')
        self.acuracia_label = QLabel('Acurácia: 0.00')
        for lbl in [self.acertos_label, self.erros_label, self.acuracia_label]:
            lbl.setFont(QFont('Arial', 12))
            stats_layout.addWidget(lbl)
        main_layout.addLayout(stats_layout)
        # Botão de reiniciar
        self.reset_button = QPushButton('Reiniciar Jogo')
        self.reset_button.setFont(QFont('Arial', 12))
        self.reset_button.clicked.connect(self.reset)
        main_layout.addWidget(self.reset_button, alignment=Qt.AlignCenter)
        self.setLayout(main_layout)
        self.atualiza_botoes()

    def estado_para_features(self):
        return pd.DataFrame([self.estado.flatten()], columns=columns)

    def checa_vitoria(self):
        tab = self.estado

        # Verifica linhas e colunas
        for i in range(3):
            if abs(tab[i, :].sum()) == 3:
                return np.sign(tab[i, 0])
            if abs(tab[:, i].sum()) == 3:
                return np.sign(tab[0, i])

        # Verifica diagonais
        if abs(np.diag(tab).sum()) == 3:
            return np.sign(tab[0, 0])
        if abs(np.diag(np.fliplr(tab)).sum()) == 3:
            return np.sign(tab[0, 2])

        # Verifica se o jogo está empatado (nenhuma célula vazia e ninguém venceu)
        if not (tab == 0).any():
            return 2  # Empate

        # Caso contrário, o jogo ainda está em andamento
        return 0


    def jogada_usuario(self, i, j):
        if self.estado[i, j] != 0 or self.fim:
            return
        self.estado[i, j] = 1
        self.atualiza_botoes()
        self.verifica_estado(jogador='usuario')
        if not self.fim:
            QTimer = QApplication.instance().thread().msleep
            QApplication.processEvents()
            self.jogada_computador()

    def jogada_computador(self):
        livres = list(zip(*np.where(self.estado == 0)))
        if livres:
            oi, oj = random.choice(livres)
            self.estado[oi, oj] = -1
            self.atualiza_botoes()
            self.verifica_estado(jogador='computador')

    def verifica_estado(self, jogador):
        vitoria = self.checa_vitoria()
        features = self.estado_para_features()
        pred = modelo.predict(features)[0]

        # Atualiza label da predição da IA com base no novo conjunto de classes
        if pred == 'positive':
            ia_msg = 'X venceu'
        elif pred == 'negative':
            # A classe "negative" pode incluir derrota de X ou empate — checa o valor real
            if vitoria == 2 or (vitoria == 0 and not (self.estado == 0).any()):
                ia_msg = 'Empate'
            else:
                ia_msg = 'O venceu'
        elif pred == 'in_progress':
            ia_msg = 'Ainda em jogo'
        else:
            ia_msg = 'Classe desconhecida pela IA'

        self.ia_label.setText(ia_msg)

        # Determina o estado real
        if vitoria == 1:
            real = 'positive'
            msg = 'Você venceu!'
        elif vitoria == -1:
            real = 'negative'
            msg = 'O venceu!'
        elif vitoria == 2:
            real = 'negative'
            msg = 'Empate!'
        else:
            real = 'in_progress'
            msg = 'Ainda em jogo'

        self.status_label.setText(msg)

        # Avalia se a IA acertou ou errou
        self.total += 1
        if pred == real:
            self.acertos += 1
            self.status_label.setText(self.status_label.text() + ' | IA acertou.')
        else:
            self.erros += 1
            self.status_label.setText(self.status_label.text() + ' | IA ERROU.')

            # Se IA errou e o jogo acabou, encerrar
            if vitoria != 0:
                self.status_label.setText(self.status_label.text() + ' Jogo encerrado: IA não detectou o fim corretamente.')
                self.fim = True
                self.desabilita_botoes()

        self.atualiza_contadores()

        # Encerrar o jogo se houve vitória ou empate
        if vitoria != 0:
            self.fim = True
            self.desabilita_botoes()

    def atualiza_botoes(self):
        for i in range(3):
            for j in range(3):
                val = self.estado[i, j]
                btn = self.botoes[i][j]
                if val == 1:
                    btn.setText('X')
                    btn.setEnabled(False)
                elif val == -1:
                    btn.setText('O')
                    btn.setEnabled(False)
                else:
                    btn.setText('')
                    btn.setEnabled(not self.fim)

    def desabilita_botoes(self):
        for i in range(3):
            for j in range(3):
                self.botoes[i][j].setEnabled(False)

    def atualiza_contadores(self):
        self.acertos_label.setText(f'Acertos: {self.acertos}')
        self.erros_label.setText(f'Erros: {self.erros}')
        acuracia = self.acertos / (self.acertos + self.erros) if (self.acertos + self.erros) > 0 else 0
        self.acuracia_label.setText(f'Acurácia: {acuracia:.2f}')

    def reset(self):
        self.estado = np.zeros((3, 3), dtype=int)
        self.fim = False
        self.status_label.setText('Ainda em jogo')
        self.ia_label.setText('Ainda em jogo')
        self.atualiza_botoes()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    janela = JogoVelha()
    janela.show()
    sys.exit(app.exec_()) 