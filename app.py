import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QTextEdit, QPushButton, QMessageBox, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal

# Import de la fonction d'initialisation depuis inference.py
from inference import setup_query_engine

class LoadEngineThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def run(self):
        try:
            # Charge les modèles et l'index
            engine = setup_query_engine()
            self.finished.emit(engine)
        except Exception as e:
            self.error.emit(str(e))

class ProcessThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, engine, observation, ordonnance):
        super().__init__()
        self.engine = engine
        self.observation = observation
        self.ordonnance = ordonnance

    def run(self):
        try:
            prompt = (
                "Tu es un assistant médical expert en pharmacologie et en médecine générale. "
                "Un médecin a fourni l'anamnèse (observation) d'un patient ainsi que son ordonnance actuelle.\n"
                "Ton rôle est d'analyser cette ordonnance à la lumière de l'anamnèse et des recommandations médicales (dont tu disposes via l'index). "
                "Tu dois proposer une ordonnance adaptée si nécessaire, en justifiant chaque modification.\n\n"
                f"Anamnèse / Observation du patient :\n{self.observation}\n\n"
                f"Ordonnance actuelle :\n{self.ordonnance}\n\n"
                "Réponds strictement en deux parties :\n"
                "1. La nouvelle ordonnance adaptée.\n"
                "2. Une explication détaillée des modifications effectuées et de ce qui a été fait selon les recommandations."
            )
            response = self.engine.query(prompt)
            self.finished.emit(str(response))
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Adaptation d'Ordonnance avec RAG")
        self.resize(800, 800)
        
        self.engine = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout(main_widget)

        # Observation
        layout.addWidget(QLabel("Observation (Anamnèse du patient) :"))
        self.observation_input = QTextEdit()
        self.observation_input.setPlaceholderText("Entrez les antécédents, les symptômes, les résultats d'examens...")
        layout.addWidget(self.observation_input)

        # Ordonnance
        layout.addWidget(QLabel("Ordonnance actuelle :"))
        self.ordonnance_input = QTextEdit()
        self.ordonnance_input.setPlaceholderText("Entrez l'ordonnance actuelle (médicaments, posologie...)")
        layout.addWidget(self.ordonnance_input)

        # Process button
        self.process_button = QPushButton("Process (Chargement des modèles...)")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.process_prescription)
        self.process_button.setMinimumHeight(40)
        self.process_button.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.process_button)

        # Barre de progression
        self.progress_bar = QProgressBar()
        # En mode indéterminé, min et max sont à 0
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Résultat
        layout.addWidget(QLabel("Nouvelle Ordonnance et Explications :"))
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        layout.addWidget(self.result_output)

        # Lancement du chargement des modèles en arrière-plan
        self.load_thread = LoadEngineThread()
        self.load_thread.finished.connect(self.on_engine_loaded)
        self.load_thread.error.connect(self.on_engine_error)
        self.load_thread.start()

    def on_engine_loaded(self, engine):
        self.engine = engine
        self.process_button.setText("Process")
        self.process_button.setEnabled(True)

    def on_engine_error(self, err_msg):
        self.process_button.setText("Erreur de chargement")
        QMessageBox.critical(self, "Erreur", f"Erreur lors du chargement des modèles : {err_msg}")

    def process_prescription(self):
        obs = self.observation_input.toPlainText().strip()
        ord_text = self.ordonnance_input.toPlainText().strip()

        if not obs or not ord_text:
            QMessageBox.warning(self, "Attention", "Veuillez remplir l'observation et l'ordonnance.")
            return

        self.process_button.setEnabled(False)
        self.process_button.setText("Génération de la nouvelle ordonnance en cours...")
        self.result_output.clear()
        
        # Afficher la barre de progression
        self.progress_bar.setVisible(True)

        self.process_thread = ProcessThread(self.engine, obs, ord_text)
        self.process_thread.finished.connect(self.on_process_finished)
        self.process_thread.error.connect(self.on_process_error)
        self.process_thread.start()

    def on_process_finished(self, result_text):
        self.result_output.setPlainText(result_text)
        self.process_button.setText("Process")
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def on_process_error(self, err_msg):
        self.process_button.setText("Process")
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Erreur", f"Erreur lors de la génération : {err_msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
