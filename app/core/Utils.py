import json
import tiktoken
from scipy import spatial
class Utils:
    @staticmethod
    def isJSON(value):
        try:
            json.loads(value)
            return True
        except Exception as ex:
            return False

    @staticmethod
    def num_tokens(text: str, model: str ):
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    
    @staticmethod
    def calculateRelatedness( x, y):
        result = 1 - spatial.distance.cosine(x, y)
        return result