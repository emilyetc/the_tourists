import json
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
import nltk
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
LOCAL_MYSQL_USER = "root"
LOCAL_MYSQL_USER_PASSWORD = "info4300"
LOCAL_MYSQL_PORT = 3306
LOCAL_MYSQL_DATABASE = "info4300"

mysql_engine = MySQLDatabaseHandler(
    LOCAL_MYSQL_USER, LOCAL_MYSQL_USER_PASSWORD, LOCAL_MYSQL_PORT, LOCAL_MYSQL_DATABASE
)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

def process_text(written_text):
    """remove stop words from the written text, transforms relevant words into a dictionary"""
    filter_out = set(stopwords.words("english"))
    tokenized = word_tokenize(written_text)
    filtered = [w for w in tokenized if not w.lower() in filter_out]
    output = defaultdict(int)
    for val in filtered:
        output[val] += 1
    return output


def process_review(review, target):
    """given a review, returns a dictionary where the keys are the strings in set target
    and the values count the number of target strings in the review"""
    output = defaultdict(int)
    review = word_tokenize(review)
    for val in review:
        val = val.lower()
        if val in target:
            output[val] += 1
    return output

def lstparser(rankinglst):
    temp = rankinglst[0][1:-1]
    output = temp.split(',')
    output = [word[1:-1].lower() for word in output]
    return output

def hotel_search(city, rankinglst, amenities, written_text):
    """city = target city, rankinglst = a string list of user's preferences with index = 0
    being the most important, amenities = target amenities, written_text =
    user's written input
    """
    rankinglst = lstparser(rankinglst)
    nltk.download('stopwords')
    nltk.download('punkt')
    # formatting user input
    written_dict = process_text(written_text)
    written_vec = [written_dict[val] for val in written_dict]

    # selecting hotels within a city (will add amenities later)
    query_sql = f"""SELECT * FROM reviews WHERE locality = '{city}'"""
    review_data = mysql_engine.query_selector(query_sql)
    review_data = review_data.all()
    # selecting rankings wtihin a city
    query_sql = f"""SELECT * FROM rankings WHERE locality = '{city}'"""
    ranking_data = mysql_engine.query_selector(query_sql)
    ranking_data = ranking_data.all()
    # print(ranking_data)
    # tracks the highest score so far
    scoretracker = defaultdict(int)
    # tracks the index of the highest score so far
    indextracker = dict()
    # print(rankinglst)
    rankingtracker = dict()
    # creating dict to index into rankings column easily
    rankingsindex = {'service': 0, 'cleanliness': 1, 'value': 2, 'location': 3, 'sleep quality': 4, 'rooms': 5}
    # calculate rankings score
    for row in range(len(ranking_data)):
        # calculate score
        score = 0
        for val in range(len(rankinglst)):
            score += (6 - val) * ranking_data[row][2 + rankingsindex[rankinglst[val]]]
        score /= 105
        key = (ranking_data[row][0], ranking_data[row][1])
        # print(key)
        rankingtracker[key] = score
    # print(rankingtracker)
    # for each item in data, calculate the review cosine similarity and add the rankings score; store it in dictionary
    for row in range(len(review_data)):
        # calculating review cosine similarity
        review_dict = process_review(review_data[row][1], list(written_dict.keys()))
        review_vec = [review_dict[val] for val in written_dict]
        denom = (norm(written_vec)*norm(review_vec))
        if denom == 0:
            cos = 0
        else:
            cos = dot(written_vec, review_vec)/denom
        key = (review_data[row][1], review_data[row][2])
        if cos + rankingtracker[key] > scoretracker[key]:
            scoretracker[key] = cos + rankingtracker[key]
            indextracker[key] = row

    # print(indextracker.keys())
    target = []
    # extract the top 3 (can change) and return
    top_n = 3
    for key, val in sorted(scoretracker.items(), key=lambda x: x[0], reverse=True)[:top_n]:
        target.append(key)
    # print(target)
    # this isn't right, only returns the columns in the review data (not sure if that's what we want)
    
    outputdata = [review_data[indextracker[key]] for key in target]


    keys = ["ratings", "title", "text", "author", "num_helpful_votes", "hotel_class", "url", "name", "locality"]
    return json.dumps([dict(zip(keys, i)) for i in outputdata])


@app.route("/")
def home():
    return render_template("base.html", title="sample html")
cities_data = []
def poss_cities():
    global cities_data
    query_sql = f"""SELECT DISTINCT locality FROM reviews"""
    res = mysql_engine.query_selector(query_sql).all()
    res = [tup[0] for tup in res]
    cities_data = res
poss_cities()

@app.route("/cities")
def cities_search():
    global cities_data
    query = request.args.get("query", "")
    matched_cities = [city for city in cities_data if query.lower() in city.lower()]
    print(jsonify(matched_cities))
    return jsonify(matched_cities)

@app.route("/find_hotels")
def find_hotels():
    city = request.args.get('city','')
    rankings = request.args.getlist('rankings')

    prompt = request.args.get('promptDescription','')
    if not city or not rankings or not prompt:
        return jsonify({"error": "Missing required parameters"}), 400
    
    resp = hotel_search(city, rankings, None, prompt)
    return resp

if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
