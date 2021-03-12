import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[2]:


app = Flask(__name__) #to create flask App
model = pickle.load(open('model.pkl', 'rb'))


# In[3]:


#to go to route/main directory of the hirearchy of web application direct
@app.route('/')
def home():
    return render_template('index.html')


# In[4]:


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0] #1 stands for donating blood; 0 stands for not donating blood
    if output == 1:
        return render_template('index.html', prediction_text='Donated blood in March 2007 as output is = {}'.format(output))
    else:
        return render_template('index.html', prediction_text='Donated blood in March 2007 as output is = {}'.format(output))


# In[5]:


if __name__ == "__main__":
    app.run(debug=True)