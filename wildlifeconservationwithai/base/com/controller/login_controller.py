from flask import redirect, render_template, request, flash

from base import app

@app.route('/')
def login_page():
    return render_template('admin/login.html')

@app.route('/admin/validate_login' , methods=['POST'])
def validate_login():
    username= 'admin'
    password='admin'

    login_username= request.form.get("emailAddress")
    login_password= request.form.get("loginPassword")

    if login_username==username:
        if login_password==password:
            return redirect('/admin/Dashboard')
        elif login_password != password:
            error_message= 'Password Is Incorrect'
            flash(error_message)
            return redirect('/login_page')
    elif login_username != username:
        error_message = "Username Is Incorrect"
        flash(error_message)
        return redirect('/login_page')








