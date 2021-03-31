import time
from celery.utils.log import get_task_logger
from app import create_celery_app

celery = create_celery_app()
logger = get_task_logger(__name__)

# from app.visao import blablabla
# from .video import blablabla

@celery.task(bind=True)
def long_task(self):
    
    for step in range(15):        
        self.update_state(state='PROGRESS', meta={"step":step})
        time.sleep(1)

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
    
    
    return {'status': 'the task have been successfully processed'}