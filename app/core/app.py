import logging
import json
from importlib import import_module
from flask import make_response

class App:
    @staticmethod
    def processRequest(request):
        try:
            urlParts = request.path.split("/")
            controllerPath = ".".join(urlParts[1:len(urlParts)-1])
            method = urlParts[len(urlParts)-1]

            getParams = request.args.to_dict() if request.args else {}
            postParams = request.form.to_dict() if request.form else {}
            params = getParams|postParams

            module = import_module("app.controllers."+controllerPath)
            ControllerClass = getattr(module, controllerPath)
            controller = ControllerClass(request)
            responseObj = getattr(controller, method)(**params)

            resp = make_response(responseObj.body)
            resp.status_code = responseObj.get_statusCode()
            resp.headers = responseObj.get_headers()

            return resp
        
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args) 

            responseBody = json.dumps({
                'success': False,
                'description' : message
            })
            resp = make_response(responseBody)
            resp.status_code = 404
            resp.headers['Content-Type'] ="application/json"

            return resp
