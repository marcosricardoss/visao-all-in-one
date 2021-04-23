""" This module contains the 'index' Blueprint which organize and
group, views related to the index endpoint of HTTP REST API.
"""

import os
from flask import url_for, request, render_template, Blueprint

HOST_ADDRESS = os.environ.get("HOST_ADDRESS", "localhost")
MEDIA_SERVICE_PORT = os.environ.get("MEDIA_SERVICE_PORT")
MEDIA_SERVICE_ADDRESS = f"http://{HOST_ADDRESS}:{MEDIA_SERVICE_PORT}"

bp = Blueprint('index', __name__, url_prefix='')


@bp.route('/', methods=['GET'])
def index():
    return render_template('index.html', 
                            url=url_for('api.get_status'),                             
                            media_service_url=MEDIA_SERVICE_ADDRESS)
