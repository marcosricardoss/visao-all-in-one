""" This module contains the 'index' Blueprint which organize and
group, views related to the index endpoint of HTTP REST API.
"""

from redis import Redis
from flask import url_for, request, render_template, Blueprint

bp = Blueprint('index', __name__, url_prefix='')

@bp.route('/', methods=['GET'])
def index():  
    return render_template('index.html', url=url_for('api.get_status'))     