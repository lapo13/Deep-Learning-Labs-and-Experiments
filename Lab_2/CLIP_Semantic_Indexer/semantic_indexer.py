import torch, os, tqdm

class semantic_indexer:
    def __init__(self, model, index_embeddings_path: str = "index_embeddings.pt"):
        self.clip_wrapper = model
        self.index_embeddings = []
        self.index_embeddings_path = index_embeddings_path
        
        
    def build_index(self, images, batch_size=4):
        if not os.path.exists(self.index_embeddings_path):
            print("Building index...")
            for step in tqdm.tqdm(range(0, len(images), batch_size), desc="Building index"):
                batch = images[step:step + batch_size]
                if images is not None:
                    image_features = self.clip_wrapper.encode(image_batch=batch)                        
                    self.index_embeddings.append(image_features)
                else:
                    raise ValueError("Either text or image must be provided to add to index.")
            torch.save(self.index_embeddings, self.index_embeddings_path)
            print(f"Saved index embeddings to disk at {os.path.abspath(self.index_embeddings_path)}")
        else:
            print("Loading index from disk...")
            self.index_embeddings = torch.load(self.index_embeddings_path)
     
    def search(self, query, top_k=5):
        print("Searching for:", query)
        if isinstance(query, str):
            query_features = self.clip_wrapper.encode(text=query)
        elif isinstance(query, torch.Tensor):
            query_features = query
        else:
            raise ValueError("Query must be either a string or a tensor.")
          
        print("Transformed Query shape:", query_features.shape) # type: ignore
        print(type(query_features))
          
        embeddings_tensor = torch.cat(self.index_embeddings, dim=0)

        print("Query features shape:", embeddings_tensor.shape)
     
        similarities = torch.nn.functional.cosine_similarity(query_features, embeddings_tensor) # type: ignore
        print("Similarities shape:", similarities.shape)
        top_k_indices = torch.topk(similarities, k=top_k)
        print("Top K indices:", top_k_indices)
        
        return top_k_indices.indices.tolist()