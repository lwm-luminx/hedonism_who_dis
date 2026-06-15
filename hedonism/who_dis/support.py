from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

API_KEY = "your_api_key_here"

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
