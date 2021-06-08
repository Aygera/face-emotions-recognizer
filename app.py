from pymongo import MongoClient
from flask import Flask, render_template
from flask import jsonify
from flask import request
from flask_pymongo import PyMongo
import emotions


app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'Cluster0'
app.config['MONGO_URI'] = "mongodb+srv://aigerim:123@cluster0.s9whk.mongodb.net/Cluster0?retryWrites=true&w=majority"

mongo = PyMongo(app)
#home page root
@app.route('/')
def root():
    return render_template('index.html')

#root to the all recordings in the MongoDB
@app.route('/all', methods=['GET'])
def get_all_results():
  fer = mongo.db.app
  output = []
  for s in fer.find():
    output.append({'image_url' : s['image_url'], 'detected_emotions' : s['detected_emotions']})
  return jsonify({'result' : output})

#root to the inserting detections result
@app.route('/', methods=['POST'])
def add_result():
  fer = mongo.db.app
  image_url = request.form["image_url"].replace("'", '')
  detected_emotions = emotions.detect_emotions(image_url)
  img_id = fer.insert({'image_url': image_url, 'detected_emotions': detected_emotions})
  new_img = fer.find_one({'_id': img_id })
  output = {'image_url' : new_img['image_url'], 'detected_emotions' : new_img['detected_emotions']}
  return render_template('index.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)