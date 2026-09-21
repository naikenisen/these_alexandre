import torch
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

def setup_query_engine():
    # 1. Configurer le modèle d'embedding
    print("Chargement du modèle d'embedding...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-large",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 2. Configurer le modèle LLM
    print("Chargement du modèle LLM...")
    Settings.llm = HuggingFaceLLM(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        tokenizer_name="microsoft/Phi-3-mini-4k-instruct",
        context_window=4096,
        max_new_tokens=1024, # Increased for detailed explanations
        generate_kwargs={"temperature": 0.1, "do_sample": True},
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16}, 
    )

    # 3. Charger l'index
    index_dir = "./multilingual-e5-large_embeddings"
    print(f"Chargement de l'index depuis {index_dir}...")
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    index = load_index_from_storage(storage_context)

    # 4. Créer le moteur de recherche
    print("Création du Query Engine...")
    return index.as_query_engine()

def main():
    try:
        query_engine = setup_query_engine()
    except Exception as e:
        print(f"Erreur lors de l'initialisation : {e}")
        return

    # 5. Boucle d'inférence
    print("\nL'index est prêt ! Vous pouvez poser vos questions. Tapez 'quit', 'q' ou 'exit' pour quitter.")
    while True:
        try:
            question = input("\nQuestion : ")
            if question.lower() in ['quit', 'q', 'exit']:
                break
            
            if not question.strip():
                continue
                
            print("Recherche et génération de la réponse en cours...")
            response = query_engine.query(question)
            print(f"\nRéponse : {response}")
                
        except KeyboardInterrupt:
            print("\nInterruption détectée. Au revoir!")
            break
        except Exception as e:
            print(f"Erreur lors de la requête : {e}")

if __name__ == "__main__":
    main()
