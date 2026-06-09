import celery
from celery import Celery
from deepface import DeepFace
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from transformers import pipeline

API_KEY = "your_api_key_here"

app = Celery('hedonism_who_dis', broker='redis://default@127.0.0.1:6379/0', result_backend='redis://default@127.0.0.1:6379/0')

PROMPT = "Write a long descriptive caption for this image in a formal tone."
pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")

def graph_client():
    # Select your transport with a defined url endpoint
    graph_transport = AIOHTTPTransport(url="http://localhost:5000/graphql",
                                       timeout=300,
                                       headers={"Authorization": f"Bearer {API_KEY}"})

    # Create a GraphQL client using the defined transport
    return Client(transport=graph_transport, execute_timeout=300)


def get_photo_url(photo_id):

    # Provide a GraphQL query
    query = gql(
        """
        query FacialRecognitionPhoto($photoId: ID!) {
          node(id: $photoId) {
            id
            ... on Photo {
              facialRecognitionUrl
            }
          }
        }
    """
    )

    query.variable_values = {"photoId": photo_id}

    result = graph_client().execute(query)
    return result['node']['facialRecognitionUrl']

@app.task()
def caption_image(photo_id):
    photo_url = get_photo_url(photo_id)

    # Build the conversation
    captioning_context = [
        {
            "role": "system",
            "content": "You are a helpful image captioner.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": photo_url},
                {"type": "text", "text": "Create a concise caption for the visually impaired."},
            ],
        },
    ]

    caption = pipe(text=captioning_context, max_new_tokens=20, return_full_text=False)[0]['generated_text']
    print(f"Generated caption: {caption}")

    description_context = [
        {
            "role": "system",
            "content": "You are a helpful image captioner.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": photo_url},
                {"type": "text", "text": "Describe the image in a professional way optimized for search"},
            ],
        },
    ]
    description = pipe(text=description_context, return_full_text=False)[0]['generated_text']
    print(f"Generated caption: {description}")

    caption_update = gql(
    """
    mutation CaptionPhoto($photoId: ID!, $caption: String!) {
      photoCaptionUpdate(id: $photoId, caption: $caption) {
        photo {
          id
        }
      }
    }
    """
    )

    caption_update.variable_values = { "photoId": photo_id, "caption": caption }
    graph_client().execute(caption_update)

@app.task()
def extract_facial_data(photo_id):
    photo_url = get_photo_url(photo_id)

    # Extract embeddings for all faces found in the image
    try:
        embedding_objects = DeepFace.represent(
            img_path=photo_url,
            expand_percentage=20,
            enforce_detection=False,
            normalization="ArcFace",
            model_name="ArcFace",         # Alternatives: "VGG-Face", "ArcFace", "OpenFace"
            detector_backend="retinaface"
        )
        print(f"Extracted {embedding_objects}")

        embedding_update = gql(
            """
            mutation FacialRecognitionPhoto($photoId: ID!, $faceObjects: [FaceDataInput!]!) {
              photoFaceUpdate(faces: $faceObjects, id: $photoId) {
                photo {
                  id
                }
              }
            }
        """
        )

        mapped_faces = [
            {
                "embedding": face['embedding'],
                "facialArea": face['facial_area'],
                "faceConfidence": face['face_confidence']
            } for face in embedding_objects
            if face['face_confidence'] > 0.9
        ]

        embedding_update.variable_values = {"photoId": photo_id, "faceObjects": mapped_faces}
        # Create a GraphQL client using the defined transport
        result = graph_client().execute(embedding_update)
        print(f"Result of Mutation => {result}")

    except Exception as e:
        print(f"An error occurred: {e}")

___all__ = ['app']

if __name__ == '__main__':
    # Equivalent to calling the celery worker command line tool
    app.worker_main(argv=['worker', '--loglevel=INFO', '--concurrency=4'])