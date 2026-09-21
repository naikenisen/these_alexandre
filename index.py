from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import torch
import gc

# Tentative de libération de la mémoire GPU (utile si l'ancien modèle est toujours en RAM)
torch.cuda.empty_cache()
gc.collect()

file_path = "/content/reco_sfd.md"

# Vérifier si le fichier existe pour éviter une erreur
if not os.path.exists(file_path):
    print(f"Le fichier {file_path} est introuvable. Veuillez le créer ou l'importer avant de continuer.")
else:
    # 1. Charger le document Markdown
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    # 2. Initialiser le parseur Markdown et extraire les nœuds
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Nombre de nœuds extraits avec MarkdownNodeParser : {len(nodes)}")

    # 3. Configurer le modèle d'embedding (multilingual-e5-large est performant et léger pour le FR)
    print("Chargement du modèle d'embedding 'intfloat/multilingual-e5-large'...")
    embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
    Settings.embed_model = embed_model
    Settings.llm = None # Défini sur None car nous faisons juste de l'indexation

    # 4. Créer l'indexation
    print("Création de l'index en cours...")
    index = VectorStoreIndex(nodes)
    print("✅ Indexation terminée avec succès !")