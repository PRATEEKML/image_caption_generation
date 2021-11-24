import os
from flask import Flask,redirect,request,render_template,url_for
from model_script import CaptionModel

app=Flask(__name__)
global model

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home')
def home_page():
    return redirect('/')

@app.route('/about', methods=['GET'])
def about_page():
    return render_template('about.html')

@app.route('/submit', methods=['POST'])
def req():
    if request.method=="POST":
        f=request.files['pic']
        image_dir='./static/user/'
        l=len(os.listdir(image_dir))+1
        fname= f'{l}_{f.filename}'
        path=image_dir+fname
        f.save(path)
        caption=model.predict_cation(path)
        return render_template('submit.html',pic=f'user/{fname}', caption=caption)

if __name__=='__main__':
    model=CaptionModel()
    if 'user' not in os.listdir('static'):
        os.mkdir('./static/user')
    else:
        for i in os.listdir('./static/user'):
            os.remove(f'./static/user/{i}')
    app.run(debug=True)