from gql import gql
from transformers import pipeline

from hedonism.who_dis.app import app
from hedonism.who_dis.support import get_photo_url, graph_client

PROMPT = "Write a long descriptive caption for this image in a formal tone."
LLAVA_PIPE = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")


@app.task()
def caption_image(photo_id):
    photo_url = get_photo_url(photo_id)
    if not photo_url:
        return

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

    caption = LLAVA_PIPE(images=captioning_context, max_new_tokens=20, return_full_text=False)[0]['generated_text']
    if caption is None:
        caption = "Unable to generate caption"
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
    description = LLAVA_PIPE(images=description_context, return_full_text=False)[0]['generated_text']
    if description is None:
        description = "Unable to generate description"
    print(f"Generated caption: {description}")

    caption_update = gql(
    """
    mutation CaptionPhoto($photoId: ID!, $caption: String!, $description: String!) {
      photoCaptionUpdate(id: $photoId, caption: $caption, description: $description) {
        photo {
          id
        }
      }
    }
    """
    )

    caption_update.variable_values = { "photoId": photo_id, "caption": caption, "description": description }
    graph_client().execute(caption_update)
