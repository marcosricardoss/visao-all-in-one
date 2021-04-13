import time
from redis import Redis
from celery.utils.log import get_task_logger
from app import create_celery_app

celery = create_celery_app()
logger = get_task_logger(__name__)
r = Redis(host='all-in-one-redis', port=6379, db=0, decode_responses=True)

# from app.visao import blablabla
# from .video import blablabla

@celery.task(bind=True)
def long_task(self):   
    """ Do a long task """   

    step = 0
    while True:
        t = "EVEN" if step % 2 == 0 else "ODD"
        self.update_state(state="PROGRESS", meta={"step":step, "type": t})
        time.sleep(1);step = step + 1

    # # step1
    # meta = {
    #     "step": 1
    # }
    # self.update_state(state='STARTED', meta=meta)
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


# start the task and send your ID to the frontend via Redis
task_id = r.get('taskid')
if not task_id:
    task = long_task.apply_async()
    r.set('taskid', task.id)