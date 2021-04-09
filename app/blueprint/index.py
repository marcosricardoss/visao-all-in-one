""" This module contains the 'index' Blueprint which organize and
group, views related to the index endpoint of HTTP REST API.
"""

from redis import Redis
from flask import url_for, request, render_template, Blueprint

bp = Blueprint('index', __name__, url_prefix='')

@bp.route('/', methods=['GET'])
def index():  
    from app.tasks import long_task, check_task_running

    running_task = check_task_running.delay()
    running_task.wait()
    task_id = None
    if running_task.result:
        task_id = str(running_task.result)
    else:
        task = long_task.apply_async()
        task_id = task.id
    
    # return 
    return render_template('index.html', url=url_for('api.get_status', task_id=task_id))     