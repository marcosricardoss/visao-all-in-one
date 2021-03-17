import os
import json
from functools import wraps
from flask import abort, request, make_response, jsonify
from jsonschema import validate, Draft7Validator, draft7_format_checker

allowed_extensions = os.environ.get('IMAGES_ALLOWED_EXTENSIONS').split("|")
allowed_extensions = { i.strip() for i in allowed_extensions}

def allowed_file(filename):
    if filename:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in allowed_extensions:
            return ext    
    return None   


def validate_file(name):   
    def decorator(f):        
        @wraps(f)
        def wrapper(*args, **kw):
            if name not in request.files:
                return make_response('No file part', 400)
            return f(*args, **kw)
        return wrapper
    return decorator