from market import app
from flask import flash, render_template, request

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        pass
    else:
        tasks = []
        return render_template('index.html', tasks=tasks)