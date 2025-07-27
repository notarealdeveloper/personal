#!/usr/bin/env python3

from flask import Flask, request
from jinja2 import Template

app = Flask(__name__)

INDEX_TEMPLATE = Template("""
    <form action="/details">
        <label for="name">Enter your name: </label>
        <input name="name" type="text"></input>
    </form>
""")

DETAILS_TEMPLATE = Template("""
    <h1>Hello {{name}}</h1>
""")

@app.route("/")
def index():
    print(request.args)
    return INDEX_TEMPLATE.render()

@app.route("/details")
def details():
    print(request.args)
    name = request.args['name']
    return DETAILS_TEMPLATE.render(name=name)

