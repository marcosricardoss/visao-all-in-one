import time
from redis import Redis
from celery.utils.log import get_task_logger
from app import create_celery_app

celery = create_celery_app()
logger = get_task_logger(__name__)

# from app.visao import blablabla
# from .video import blablabla

@celery.task()
def check_task_running(): # pragma: no cover
    """ Get the current long_task ID. """

    i = celery.control.inspect(['celery@all-in-one-worker'])
    workers = i.active()      
    for task in workers['celery@all-in-one-worker']:
        if workers['celery@all-in-one-worker'][0]["name"] == 'app.tasks.long_task':
            return workers['celery@all-in-one-worker'][0]['id']            
    
    return None

@celery.task(bind=True)
def long_task(self):   
    """ Do a long task """
    
    step = 0
    while True:
        state = "EVEN" if step % 2 == 0 else "ODD"
        self.update_state(state=state, meta={"step":step})
        time.sleep(1);step = step + 1

    # # step1
    # meta = {
    #     "step": 1
    # }
    # self.update_state(state='STATED', meta=meta)
    # time.sleep(5) # code
    
    # # step2
    # meta = {
    #     "step": 2
    # }
    # self.update_state(state='PROGRESS', meta=meta)
    # time.sleep(2) # code    
    
    # # step3
    # meta = {
    #     "step": 3
    # }
    # self.update_state(state='FINISHED', meta=meta)
    # time.sleep(2) # code        
    
    return {'status': 'CONCLUDED'}