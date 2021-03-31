"""This module contains class whose instances will be used to
load the settings according to the running environment. """


import os
import datetime

class Default():
    """Class containing the default settings for all environments."""

    TESTING = False    
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND')
    CELERY_TASK_LIST = ['app.tasks']    

class Production(Default):
    """Class containing the settings of the production environment."""

    pass
    

class Staging(Default):
    """Class containing the settings of the staging environment."""

    pass

class Development(Default):
    """Class containing the settings of the development environment."""

    DEBUG = True   
