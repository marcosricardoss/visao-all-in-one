""" This module contains the API Blueprint which organize and
group, views related to the index endpoint of HTTP REST API.
"""

from redis import Redis
from flask import Blueprint, jsonify, make_response, request, url_for

bp = Blueprint('api', __name__, url_prefix="/api/v1.0/task")
r = Redis(host='all-in-one-redis', port=6379, db=0, decode_responses=True)

@bp.route('/', methods=['POST'])
def press_button():               
    from app.tasks import long_task 
    # checking for a running task
    task_id = r.get('taskid')
    if task_id: 
        task = long_task.AsyncResult(task_id)
        if task.state == "PENDING" or task.state=="PROGRESS":
            return make_response(jsonify({
                "msg": "there is another task running",                
                "task": task.id
            }), 429)  
    
    # creating a new task
    task = long_task.apply_async()     
    return make_response(jsonify({
        "state": task.state,
        "id": task.id,
    }), 200)    


@bp.route('/', methods=['GET'])
def get_status():         
    # checking for a running task
    task_id = r.get('taskid')
    if not task_id:
        return make_response(jsonify({
            "msg": "there is no task running",            
        }), 400)      

    # retreiving the task states
    from app.tasks import long_task
    task = long_task.AsyncResult(task_id)
    return make_response(jsonify({
        "task_id": task.id,
        "task_state": task.state,
        "data": task.info
    }), 200)


@bp.route('/inspect/<string:task_id>', methods=['GET'])
def inspect(task_id):         
    from app.tasks import long_task
    
    # checking for a running task    
    if not task_id:
        return make_response(jsonify({
            "msg": "there is no task running",
        }), 400)      

    # retreiving the task states
    task = long_task.AsyncResult(task_id)
    return make_response(jsonify({
        "state": str(task.state),        
        "info": task.info
    }), 200)     
