import os

from flask import Flask
from celery import Celery

from app.util import load_config
from app.blueprint import index, api

def create_celery_app(app: Flask = None):
    """Create a celery application

    Parameters:
    flask.app.Flask: A application instance    

    Returns: 
    celery.Celery = A celery instance

    """

    app = app or create_app("celery")
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


def create_app(package_name: str = None, test_config: dict = {}) -> Flask:
    """This function is responsible to create a Flask instance according
    a previous setting passed from environment. In that process, it also
    initialise the database source.

    Parameters:
    test_config (dict): settings coming from test environment

    Returns:
    flask.app.Flask: The application instance
    """

    package_name = package_name if package_name else __name__
    app = Flask(package_name, instance_relative_config=True)
    load_config(app, test_config)

    # blueprints    
    app.register_blueprint(index.bp)        
    app.register_blueprint(api.bp)

    return app
