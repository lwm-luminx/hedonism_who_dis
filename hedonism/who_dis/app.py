from celery import Celery


app = Celery('hedonism_who_dis', broker='redis://default@127.0.0.1:6379/0', result_backend='redis://default@127.0.0.1:6379/0')
app.autodiscover_tasks(["hedonism.who_dis"], force=True)


___all__ = ['app']

if __name__ == '__main__':
    # Equivalent to calling the celery worker command line tool
    app.worker_main(argv=['worker', '--loglevel=INFO', '--concurrency=4'])