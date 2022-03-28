from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy  import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from temp_main import *
import os
import sqlite3


PEOPLE_FOLDER = os.path.join('static', 'people')
DETECT_FOLDER = os.path.join('static', 'detections')

global text_arr
text_arr = []

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Thisissupposedtobesecret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////database.db'
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
app.config['FRAME_FOLDER'] = DETECT_FOLDER

bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=40)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=4, max=20)])
    remember = BooleanField('remember me')

class RegisterForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=40)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=4, max=20)])


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/user', methods=['GET', 'POST'])
def user():
    return render_template('user.html')


@app.route('/error_l', methods=['GET', 'POST'])
def error_l():
    return render_template('error_l.html')


@app.route('/error_s', methods=['GET', 'POST'])
def error_s():
    return render_template('error_s.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('user'))

        # return '<h1>Invalid username or password</h1>'
        text = 'Invalid username or password'
        return render_template('error_l.html', content=text)

    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        try:
            db.session.add(new_user)
            db.session.commit()
            # return '<h1> New user has been created! </h1>'
            text = 'New user has been created!'
            return render_template('succ_s.html', content=text)
        except:
            # return '<h1> User already exists! </h1>'
            text = 'User already exists!'
            return render_template('error_s.html', content=text)

        #return '<h1>' + form.username.data + ' ' + form.email.data + ' ' + form.password.data + '</h1>'

    return render_template('signup.html', form=form)


@app.route("/image", methods=['GET','POST'])
def image():
    if request.method != 'POST':
        return render_template('upload-image.html')

    try:
        a = request.form['name_data']
        b = request.files['image_data']

        if not a or b.filename == '':
            text = 'Invalid name or image! Please provide proper inputs above!'
            return render_template('upload-image.html', content=text)

        c = secure_filename(f'{a}.jpg')
        c = c.replace('_', ' ')
        b.save(os.path.join(app.config['UPLOAD_FOLDER'], c))

        text = 'New user image has been saved in database!'
        return render_template('upload-image.html', content=text)

    except:
        text_inv = 'Invalid input! Please provide proper inputs above!'
        return render_template('upload-image.html', content=text_inv)



@app.route("/video", methods=['GET','POST'])
@login_required
def video():
    global text_arr
    if request.method != 'POST':
        return render_template('upload-video.html')
    try:
        a = request.form['name_data']
        b = request.files['video_data']

        if a and b.filename == '':
            text = 'Invalid video input! Please upload a proper video!'
            return render_template('upload-video.html', content=text)
        
        if not a and b.filename != '':
            text = 'Invalid person name! Please Register First! or Provide a registered person name!'
            return render_template('upload-video.html', content=text)

        b.save(os.path.join(secure_filename('uploaded_video.mp4')))
        text_arr = main(a)
        return redirect(url_for('result'))

    except:
        # return '<h1>Invalid username or password</h1>'
        text_inv = 'Invalid username or input video! Please provide the inputs correcly!'
        return render_template("upload-video.html", content=text_inv)


@app.route("/result")
@login_required
def result():
    global text_arr
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], text_arr[0])
    print(full_filename)
    try:
        # print(text_arr[2])
        frame_filename = os.path.join(app.config['FRAME_FOLDER'], text_arr[2])
        frame_filename2 = os.path.join(frame_filename, text_arr[3])
        print(frame_filename2)
        return render_template("result.html", user_image1 = full_filename, user_image2 = frame_filename2, content=text_arr[1], content0='\n--- See the results below! ---\n', content1='\nInput image (saved in database):\n', content2='\nDetected frame from the video:\n')
    except:
        return render_template("result.html", user_image1 = full_filename, content=text_arr[1])
    # if request.method == 'POST':
    #     return render_template('upload-video.html')
    # return render_template('result.html')



@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.username)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
