import torch
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

def main():
    # 1. Configurer le modèle d'embedding (doit être le même que celui utilisé pour créer l'index)
    print("Chargement du modèle d'embedding...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/e5-mistral-7b-instruct",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 2. Configurer le modèle LLM (Llama-3.1-8B-Instruct)
    print("Chargement du modèle LLM...")
    # NOTE: Llama 3.1 nécessite d'avoir accepté les conditions sur HuggingFace et d'être authentifié 
    # via `huggingface-cli login` ou d'utiliser une variable d'environnement HF_TOKEN.
    Settings.llm = HuggingFaceLLM(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        context_window=8192,
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.1, "do_sample": True},
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16}, 
        # Décommentez la ligne ci-dessous si vous avez des limites de VRAM pour charger en 8-bit
        # model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}, 
    )

    # 3. Charger l'index depuis le dossier
    index_dir = "./e5-mistral-7b-instruct_embeddings"
    print(f"Chargement de l'index depuis {index_dir}...")
    try:
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
    except Exception as e:
        print(f"Erreur lors du chargement de l'index : {e}")
        return

    # 4. Créer le moteur de recherche (Query Engine)
    print("Création du Query Engine...")
    query_engine = index.as_query_engine()

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
            
            # Affichage optionnel des sources (documents utilisés pour répondre)
            # print("\n--- Sources ---")
            # for node in response.source_nodes:
            #     print(f"- Score: {node.score:.3f} | Text: {node.text[:200]}...")
                
        except KeyboardInterrupt:
            print("\nInterruption détectée. Au revoir!")
            break
        except Exception as e:
            print(f"Erreur lors de la requête : {e}")

if __name__ == "__main__":
    main()
