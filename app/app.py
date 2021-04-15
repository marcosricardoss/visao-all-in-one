import os
import random
import collections

from celery import Celery
from waitress import serve
from flask import Flask, jsonify, redirect, render_template, request, url_for

##
## FLASK CONFIGURATION
##

app = Flask(__name__)
app.config['TESTING'] = False  
app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL')
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_RESULT_BACKEND')
app.config['CELERY_TASK_LIST'] = ['app.tasks']    

##
## CELERY APP SERVER 
##

def create_celery_app():
    """Create a celery application
    Parameters:
    flask.app.Flask: A application instance    
    Returns: 
    celery.Celery = A celery instance
    """

    global app
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL'],
        include=app.config['CELERY_TASK_LIST'],        
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task): # pragma: no cover
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


##
## ROUTES
##


@app.route('/', methods=['GET'])
def index():    
    return render_template('index.html', url=url_for('get_data'))


d = collections.deque(["#17a2b8", "#dd4b39", "#28a745"]) # DEBUG
step = 0 # DEBUG
@app.route('/data', methods=['GET'])
def get_data():           
    global d; # DEBUG
    global step
    step = 0 if step == 15 else step + 1
    colors = list(d)    
    return jsonify({
        "step": step,
        "colors": colors
    })         

##
## RUNNING THE SERVER 
##

if __name__ == '__main__':        
    if os.environ.get('FLASK_ENV') == 'development' or os.environ.get('FLASK_ENV') == 'testing':
        app.config['DEBUG'] = True    
        app.run(host='0.0.0.0', port=int(os.environ.get('FLASK_PORT')))
    else:                             
        app.config['DEBUG'] = False       
        serve(app, host='0.0.0.0', port=os.environ.get('FLASK_PORT'))