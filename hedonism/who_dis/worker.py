from celery import Celery
from deepface import DeepFace
import psycopg2
from sklearn.cluster import DBSCAN
import numpy as np
import json
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
          photo(id: $photoId) {
            id
            facialRecognitionUrl
          }
        }
    """
    )

    query.variable_values = {"photoId": photo_id}

    result = graph_client().execute(query)
    return result['photo']['facialRecognitionUrl']

@app.task
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
                {"type": "text", "text": "Describe the contents of this image in a professional way optimized for search"},
            ],
        },
    ]
    caption_result = pipe(text=captioning_context)
    caption = caption_result[0]['generated_text'][2]['content']
    print(f"Generated caption: {caption}")

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
    result = graph_client().execute(caption_update)


@app.task
def cluster_faces(date_grouping):
    # 1. Connect to PostgreSQL and fetch embeddings
    with psycopg2.connect("dbname=hedonism_bot_development host=localhost port=5432") as conn:
        with conn.cursor() as cur:
            if date_grouping is None:
                cur.execute("SELECT photo_people.id, arc_face_embedding FROM photo_people WHERE photo_people.arc_face_embedding IS NOT NULL")
            else:
                cur.execute("SELECT photo_people.id, arc_face_embedding FROM photo_people INNER JOIN photos ON photo_people.photo_id = photos.id WHERE photos.folder_date = %s", (date_grouping,))
            data = cur.fetchall()

            # 2. Separate IDs and Vectors
            ids = [row[0] for row in data]
            embeddings = np.array([json.loads(row[1]) for row in data])

            # 3. Apply DBSCAN
            # eps: maximum distance between two samples for them to be considered as in the same neighborhood
            # min_samples: number of samples in a neighborhood for a point to be considered as a core point
            clustering = DBSCAN(eps=0.3, min_samples=1, metric='cosine').fit(embeddings)
            labels = clustering.labels_

            # 4. Update the database with cluster assignments
            update_data = [ [ int(label), id_] for label, id_ in zip(labels, ids)   ]

    return list(update_data)

@app.task
def extract_facial_data(photo_id):
    photo_url = get_photo_url(photo_id)

    # Extract embeddings for all faces found in the image
    try:
        embedding_objects = DeepFace.represent(
            img_path=photo_url,
            enforce_detection=False,
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