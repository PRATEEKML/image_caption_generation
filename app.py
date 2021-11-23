from flask import Flask,redirect,request,render_template,url_for
from model_script import gen_caption

app=Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/home')
def home_page():
	return redirect('/')

@app.route('/submit', methods=['POST'])
def req():
	if request.method=="POST":
		f=request.files['pic']
		f.save('./static/'+f.filename)
		caption='Seems like , '+gen_caption('./static/'+f.filename)
		return render_template('submit.html',pic=f.filename,caption=caption)

if __name__=='__main__':
	app.run(debug=True)