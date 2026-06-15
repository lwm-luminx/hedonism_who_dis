from deepface import DeepFace
from gql import gql

from hedonism.who_dis.app import app
from hedonism.who_dis.support import get_photo_url, graph_client

@app.task()
def extract_facial_data(photo_id):
    photo_url = get_photo_url(photo_id)
    if not photo_url:
        return

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
