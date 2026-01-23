import time
import unittest
import grpc
import superbuilder_service_pb2_grpc as sb_grpc
from helpers.chat import warmup, get_chat_history, generate_random_session_id, init_chat_session, set_chat_request, get_chat_response, remove_session

class TestChatHelpers(unittest.TestCase):

    def setUp(self):
        time.sleep(10)  # Allow time for the service to start
        # Connect to the actual service
        self.grpc_address = 'localhost:5006'
        self.channel = grpc.insecure_channel(self.grpc_address)
        self.stub = sb_grpc.SuperBuilderStub(self.channel)

    def tearDown(self):
        # Close the gRPC channel
        self.channel.close()

    def test_warmup(self):
        try:
            warmup(self.stub)
            print("Warmup completed successfully.")
        except grpc.RpcError as e:
            self.fail(f"gRPC error during test_warmup: {e.details()}")
        except Exception as e:
            self.fail(f"Error during test_warmup: {e}")

    def test_get_chat_history(self):
        try:
            result = get_chat_history(self.stub)
            self.assertIsInstance(result, list, "Chat history is not a list")
        except grpc.RpcError as e:
            self.fail(f"gRPC error during test_get_chat_history: {e.details()}")
        except Exception as e:
            self.fail(f"Error during test_get_chat_history: {e}")

    def test_generate_random_session_id(self):
        try:
            chat_history = get_chat_history(self.stub)
            result = generate_random_session_id(chat_history)
            self.assertNotIn(result, [session["sid"] for session in chat_history], "Generated session ID is not unique")
            self.assertIsInstance(result, int, "Generated session ID is not an integer")
        except Exception as e:
            self.fail(f"Error during test_generate_random_session_id: {e}")

    def test_init_chat_session(self):
        try:
            result = init_chat_session(self.stub)
            self.assertIsInstance(result, int, "Session ID is not an integer")
        except grpc.RpcError as e:
            self.fail(f"gRPC error during test_init_chat_session: {e.details()}")
        except Exception as e:
            self.fail(f"Error during test_init_chat_session: {e}")

    def test_set_chat_request(self):
        try:
            prompt = "Hello, world!"
            session_id = init_chat_session(self.stub)
            response_iterator = set_chat_request(self.stub, prompt, session_id=session_id)
            self.assertIsNotNone(response_iterator, "Response iterator is None")
        except grpc.RpcError as e:
            self.fail(f"gRPC error during test_set_chat_request: {e.details()}")
        except Exception as e:
            self.fail(f"Error during test_set_chat_request: {e}")

    def test_get_chat_response(self):
        try:
            prompt = "Hello, world!"
            session_id = init_chat_session(self.stub)
            response_iterator = set_chat_request(self.stub, prompt, session_id=session_id)
            result = get_chat_response(response_iterator, verbose=False)
            self.assertIsInstance(result, str, "Chat response is not a string")
            print("Chat response retrieved successfully:", result)
        except grpc.RpcError as e:
            self.fail(f"gRPC error during test_get_chat_response: {e.details()}")
        except Exception as e:
            self.fail(f"Error during test_get_chat_response: {e}")

    def test_remove_session(self):
        try:
            session_id = init_chat_session(self.stub)
            remove_session(self.stub, session_id)
            print(f"Session {session_id} removed successfully.")
        except grpc.RpcError as e:
            self.fail(f"gRPC error during test_remove_session: {e.details()}")
        except Exception as e:
            self.fail(f"Error during test_remove_session: {e}")


if __name__ == '__main__':
    unittest.main()