class Controller:
    def __init__(self,request):
        self.req = request

    def get_req(self):
        return self.req
    
    def set_req(self,value):
        self.req = value