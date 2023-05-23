from app.core.Utils import Utils
class Response:
    def __init__(self, statusCode=200, statusMessage="", headers={}, body=""):
        self.statusCode = statusCode
        self.statusMessage = statusMessage
        self.body = body

        if Utils.isJSON(body):
            self.headers = headers | {"Content-Type": "application/json" } | {"Access-Control-Allow-Origin":"*"}
        else:
            self.headers = headers | { "Content-Type": "text/plain"  } | {"Access-Control-Allow-Origin":"*"}

    
    def get_statusCode(self):
        return self.statusCode
    
    def set_statusCode(self,value):
        self.statusCode = value
    
    def get_statusMessage(self):
        return self.statusMessage
    
    def set_statusMessage(self,value):
        self.statusMessage = value
    
    def get_headers(self):
        return self.headers
    
    def set_headers(self,value):
        self.headers = value

    def get_body(self):
        return self.body
    
    def set_body(self,value):
        self.body = value
    