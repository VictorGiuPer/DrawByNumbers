from flask import Flask, render_template, request, redirect, url_for
import os

from src import main


app = Flask(__name__)

UPLOAD_FOLDER = 'data/'  # Point to the 'data' folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)