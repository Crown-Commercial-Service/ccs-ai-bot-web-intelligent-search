from uvicorn.workers import UvicornWorker

class UvicornH11Worker(UvicornWorker):
    CONFIG_KWARGS = {"loop": "asyncio", "http": "h11"}

# Import the ASGI application
from app import app as asgi_app

# Define a callable WSGI object
def application(environ, start_response):
    # This is a hack for Azure App Service which expects a WSGI application
    # but we're using an ASGI application (FastAPI)
    raise RuntimeError(
        "You should be using the Uvicorn worker. "
        "Try specifying --workers-class=uvicorn.workers.UvicornWorker or use --wsgi wsgi:asgi_app"
    )