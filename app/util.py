import os
from flask import Flask

def load_config(app: Flask, test_config) -> None:
    """Load the application's config.

    Parameters:
    app (flask.app.Flask): The application instance Flask that'll be running
    test_config (dict):
    """

    if test_config:
        if test_config.get('TESTING'):
            app.config.from_mapping(test_config)
        else:
            app.config.from_object(
                f'app.config.{test_config.get("FLASK_ENV").capitalize()}')
    else:
        app.config.from_object(
            f'app.config.{os.environ.get("FLASK_ENV").capitalize()}')