import grpc
from google.protobuf.json_format import MessageToDict
import chat_pb2
import chat_pb2_grpc

class DatasakiAIClient(object):
    def __init__(self,user_id):
        self.channel = grpc.insecure_channel("172.18.0.8:50061")
        self.stub = chat_pb2_grpc.ChatServiceStub(self.channel)
        self.user_id = user_id

    def run_ai_call(self, question:str,system_prompt:str,industry:str=None):
        user_query = chat_pb2.UserQuery(
            question=question,
            system_prompt=system_prompt,
            user_id=str(self.user_id),
            industry=industry
        )
        try:
            # Send the request
            response = self.stub.ChatWithGroq(user_query)
            return response.response

        except grpc.RpcError as rpc_error:
            print(rpc_error.details())
            return {
                "chat_status": rpc_error.details()
            }


