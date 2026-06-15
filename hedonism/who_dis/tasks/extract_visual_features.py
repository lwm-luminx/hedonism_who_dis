import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from hedonism.who_dis.app import app
from hedonism.who_dis.support import get_photo_url, graph_client

model_name = "google/vit-large-patch32-224-in21k"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

@app.task()
def extract_visual_features(photo_id):
    # 2. Load and prepare your image
    # Replace with your own image path
    image_path = get_photo_url(photo_id)
    image = Image.open(image_path).convert("RGB")

    # 3. Preprocess the image (resizes to 224x224 and normalizes)
    inputs = processor(images=image, return_tensors="pt")

    # 4. Extract embeddings without computing gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. Extract the embedding from the last hidden state
    # ViT outputs the [CLS] token at index 0, which represents the whole image
    last_hidden_states = outputs.last_hidden_state
    image_embedding = last_hidden_states[:, 0, :]

    # Output shape will be: torch.Size([1, 1024])
    print("Embedding Shape:", image_embedding.shape)
    print("Embedding Tensor:\n", image_embedding)

    embedding_update = gql(
        """
        mutation PhotoFeature($photoId: ID!, $embedding: [Float!]!) {
          photoFeaturesUpdate(embedding: $embedding, id: $photoId) {
            photo {
              id
            }
          }
        }
    """
    )

    embedding_update.variable_values = {"photoId": photo_id, "embedding": embedding}
    result = graph_client().execute(embedding_update)
    print(f"Result of Mutation: {result}")

