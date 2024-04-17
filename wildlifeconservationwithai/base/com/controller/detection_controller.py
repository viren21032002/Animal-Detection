import os

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename, redirect

from base import app
from base.com.vo.detection_vo import DetectionVO
from base.com.dao.detection_dao import DetectionDAO
from base.com.services.detection_services import run

INPUT_FOLDER="base/static/adminResources/input_videos/"
OUTPUT_FOLDER="base/static/adminResources/output_video/"
app.config['INPUT_FOLDER']=INPUT_FOLDER
app.config['OUTPUT_FOLDER']=OUTPUT_FOLDER

@app.route('/admin/loadDashboard')
def adminHomapage():
    detection_dao = DetectionDAO()
    data = detection_dao.view_detection_images()
    lionCount=len(data)
    return render_template('admin/index.html',lionCount=lionCount)

@app.route('/admin/Dashboard')
def adminDashboard():
    return render_template('admin/index.html')

@app.route('/admin/addDetection')
def addDetection():
    return render_template('admin/addDetection.html')

@app.route('/admin/uploadDetection', methods=['post'])
def uploadDetection():
    input_file = request.files.get('InputFile')
    input_file_name = secure_filename(input_file.filename)
    input_file_path = os.path.join(app.config['INPUT_FOLDER'],
                                       input_file_name)
    input_file.save(input_file_path)

    # detection on File Uploaded By User
    source = r"{}".format(input_file_path)

    run(source=source)
    output_file_path =  os.path.join(app.config['OUTPUT_FOLDER'],
                                        input_file_name)

    detection_vo = DetectionVO()
    detection_dao = DetectionDAO()

    detection_vo.file_name = input_file_name
    detection_vo.input_file_path = input_file_path.replace('base', '')
    detection_vo.output_file_path = output_file_path.replace('base', '')

    detection_dao.add_detection(detection_vo)
    return redirect ('/admin/viewDetection')

@app.route('/admin/viewDetection')
def viewDetection():
    detection_dao=DetectionDAO()
    data=detection_dao.view_detection()
    return render_template('admin/viewDetection.html' , data=data )

@app.route('/admin/viewImages')
def viewImages():
    detection_dao=DetectionDAO()
    data=detection_dao.view_detection_images()
    return render_template('admin/detectionImages.html', data=data)
