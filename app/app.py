import os
import random
import collections
from waitress import serve
from flask import Flask, request, render_template, redirect, jsonify

##
## FLASK CONFIGURATION
##

app = Flask(__name__)
app.config['TESTING'] = False  

@app.route('/', methods=['GET'])
def index():    
    return render_template('index.html', url="http://localhost:8000/data")

d = collections.deque(["#777", "green", "#777"]) # DEBUG
@app.route('/data', methods=['GET'])
def get_data():           
    global d; d.rotate() # DEBUG
    colors = list(d)    
    return jsonify({
        "colors": colors
    })         

##
## RUNNING THE SERVER 
##

if __name__ == '__main__':        
    if os.environ.get('FLASK_ENV') == 'development' or os.environ.get('FLASK_ENV') == 'testing':
        app.config['DEBUG'] = True    
        app.run(host='0.0.0.0', port=int(os.environ.get('FLASK_PORT')))
    else:                             
        app.config['DEBUG'] = False       
        serve(app, host='0.0.0.0', port=os.environ.get('FLASK_PORT'))