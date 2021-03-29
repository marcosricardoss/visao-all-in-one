""" This module contains the API Blueprint which organize and
group, views related to the index endpoint of HTTP REST API.
"""

from flask import Blueprint, jsonify, make_response, request, url_for

bp = Blueprint('api', __name__, url_prefix="/api/v1.0/task")
task_id = None


@bp.route('/', methods=['POST'])
def longtask():               
    from app.tasks import long_task

    if request.method == 'POST':
        task = long_task.apply_async()
        task_id=task.id        
        return make_response(jsonify({
            "taskID": str(task.id),
        }), 200)      

@bp.route('/<string:task_id>', methods=['GET'])
def get_data(task_id):         
    from app.tasks import long_task
    
    task = long_task.AsyncResult(task_id)
    return make_response(jsonify({
        "task_state": str(task.state),
        "step": task.info.get("step")
    }), 200)
    
