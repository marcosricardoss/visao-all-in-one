import time
from celery.utils.log import get_task_logger
from app import create_celery_app

celery = create_celery_app()
logger = get_task_logger(__name__)

@celery.task(bind=True)
def long_task(self):
    for step in range(15):        
        self.update_state(state='PROGRESS', meta={"step":step})
        time.sleep(1)
    return {'status': 'the task have been successfully processed'}