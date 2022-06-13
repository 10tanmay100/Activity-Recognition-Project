
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import warnings
from flask_cors import CORS, cross_origin
warnings.filterwarnings("ignore")
# os.chdir("E:/ml project/classification project/Activity Recognition/logger")
import pandas as pd
app = Flask(__name__,template_folder="templates")
model_1 = pickle.load(open('3_RFID.pkl', 'rb'))
scale_1=pickle.load(open("3_RFID_Scale.pkl","rb"))
model_2=pickle.load(open('4_RFID.pkl',"rb"))
scale_2=pickle.load(open("4_RFID_Scale.pkl","rb"))
# os.chdir("E:/ml project/classification project/Activity Recognition/templates")

@app.route('/',methods=['POST', 'GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    '''
    For rendering results on HTML GUI
    '''
    a=request.form.get("time_seconds")
    b=request.form.get("Accelaration in G Frontal")
    c=request.form.get("Accelaration in G Vertical")
    d=request.form.get("Accelaration in G Lateral")
    e=request.form.get("RSSI")
    f=request.form.get("Phase")
    g=request.form.get("Frequency")
    h=request.form.get("ID")

    df=pd.DataFrame({"time_seconds":[a],"Accelaration in G Frontal":[b],"Accelaration in G Vertical":[c],"Accelaration in G Lateral":[d],"RSSI":[e],"Phase":[f],"Frequency":[g]})
    if h==3:
        transformed_data=scale_1.transform(df)
    else:
        transformed_data=scale_2.transform(df)
    a=transformed_data[0][0]
    b=transformed_data[0][2]
    c=transformed_data[0][3]
    d=transformed_data[0][4]
    e=transformed_data[0][5]
    f=transformed_data[0][6]
    ans=np.hstack([a,b,c,d,e,f]).reshape(1,-1)
    if h==3:
        prediction = model_1.predict(ans)
    else:
        prediction = model_2.predict(ans)

    output = prediction
    if output==1:
        ans="sit on bed"
    elif output==2:
        ans="sit on chair"
    elif output==3:
        ans="lying"
    else:
        ans="ambulating"

    return render_template('res.html', prediction='Activity Label is {}'.format(ans))
if __name__=='__main__':
       app.run(debug=True)

