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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
import numpy as np

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
# mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)
nltk.download('stopwords')
nltk.download('punkt')
def process_text(written_text):
    """remove stop words from the written text, transforms relevant words into a dictionary"""
    filter_out = set(stopwords.words("english"))
    # negative stop words needed for sentiment analysis
    negative_terms = set(["no", "not", "none", "nor", "never", "shouldn't", "won't", "doesn't", "isn't", "wouldn't", "without"])
    filter_out = filter_out.difference(negative_terms)
    tokenized = word_tokenize(written_text)
    filtered = [w for w in tokenized if not w.lower() in filter_out]
    output = defaultdict(int)
    #If a term is preceded by a negative term, reduce term frequency by one
    #Note: negative terms are not included as keys in the output dictionary
    for i in range(len(filtered)):
        term = filtered[i].lower()
        prev_term = None if i==0 else filtered[i-1].lower()
        if (i==0 and term not in negative_terms):
            output[term] += 1
        elif (term not in negative_terms):
            if (prev_term in negative_terms):
                output[term] -= 1
            else:
                output[term] += 1
    for term, freq in output.items():
        output[term] = freq if freq >= 0 else 0
    return output

def process_review(review, target):
    """given a review, returns a dictionary where the keys are the strings in set target
    and the values count the number of target strings in the review, and a modified review
    text with the matching words bolded"""
    negative_terms = set(["no", "not", "none", "nor", "never", "shouldn't", "won't", "doesn't", "isn't", "wouldn't", "without"])
    output = defaultdict(int)
    review_words = word_tokenize(review)
    modified_review = []
    for i in range(len(review_words)):
        term = review_words[i].lower()
        prev_term = None if i==0 else review_words[i-1].lower()
        if(term in target):
            if(i==0 and term not in negative_terms):
                output[term] += 1
                modified_review.append(f"<b>{term}</b>")
            elif(prev_term not in negative_terms):
                output[term] += 1
                modified_review.append(f"<b>{term}</b>")
            elif(prev_term in negative_terms):
                output[term] -= 1
                modified_review.append(f"<b>{term}</b>")
        else:
            modified_review.append(term)
    for term, freq in output.items():
        output[term] = freq if freq >= 0 else 0
    modified_review = " ".join(modified_review)
    return output, modified_review

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
    # formatting user input
    written_dict = process_text(written_text)
    written_vec = [written_dict[val] for val in written_dict]

    # selecting hotels within a city (will add amenities later)
    query_sql = f"""SELECT * FROM reviews WHERE locality = '{city}'"""
    review_data = mysql_engine.query_selector(query_sql)
    review_data = review_data.all()
    review_data = [[str(col) for col in row] for row in review_data]
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
        review_dict, modified_review = process_review(review_data[row][0], list(written_dict.keys()))
        review_data[row][0] = modified_review
        review_vec = [review_dict[val] for val in written_dict]
        denom = (norm(written_vec)*norm(review_vec))
        if denom == 0:
            cos = 0
        else:
            cos = dot(written_vec, review_vec)/denom
        key = (review_data[row][1], review_data[row][2])
        if cos + rankingtracker[key] > scoretracker[key]: #cosine sim seems irrelevant, so adding a multiplier
            scoretracker[key] = cos + rankingtracker[key]
            indextracker[key] = row

    # print(indextracker.keys())
    target = []
    # extract the top 3 (can change) and return
    top_n = 3
    '''for key, val in sorted(scoretracker.items(), key=lambda x: x[0], reverse=True)[:top_n]:
        target.append(key)'''
    # print(target)
    # this isn't right, only returns the columns in the review data (not sure if that's what we want)
    target = sorted(scoretracker, key=scoretracker.get, reverse=True)[:top_n]
    outputdata = [review_data[indextracker[key]] for key in target]


    keys = ["ratings", "title", "text", "author", "num_helpful_votes", "hotel_class", "url", "name", "locality"]
    return json.dumps([dict(zip(keys, i)) for i in outputdata])

def attraction_2(city, written_text):
    written_dict = process_text(written_text)
    written_vec = [written_dict[val] for val in written_dict]

    query_sql = f"""SELECT * FROM attractions WHERE City = '{city}'"""
    attraction_data = mysql_engine.query_selector(query_sql)
    attraction_data = attraction_data.all()
    scoretracker = defaultdict(int)
    indextracker = dict()

    for index, (city, location_name, description) in enumerate(attraction_data):
        attraction_dict = process_review(description, list(written_dict.keys()))
        attraction_vec = [attraction_dict[val] for val in written_dict]
        denom = (norm(written_vec) * norm(attraction_vec))
        cos = 0 if denom == 0 else dot(written_vec, attraction_vec) / denom
        key = (location_name, description)
        if cos > scoretracker[key]:
            scoretracker[key] = cos
            indextracker[key] = index

    target = sorted(scoretracker, key=scoretracker.get, reverse=True)[:3]
    outputdata = [attraction_data[indextracker[key]] for key in target]
    keys = ["City", "Location_Name", "Description"]
    return json.dumps([dict(zip(keys, data)) for data in outputdata])
    pass
def attraction_search(city, written_text):
    """city = target city, written_text = user's written input
    """
    # formatting user input
    written_dict = process_text(written_text)
    written_vec = [written_dict[val] for val in written_dict]

    # selecting amenities within a city
    query_sql = f"""SELECT * FROM attractions WHERE City = '{city}'"""
    attraction_data = mysql_engine.query_selector(query_sql)
    attraction_data = attraction_data.all()

    # tracks the highest score so far
    scoretracker = defaultdict(int)
    # tracks the index of the highest score so far
    indextracker = dict()
    for row in range(len(attraction_data)):
        attraction_dict = process_review(attraction_data[row][0], list(written_dict.keys()))
        attraction_vec = [attraction_dict[val] for val in written_dict]
        denom = (norm(written_vec)*norm(attraction_vec))
        if denom == 0:
            cos = 0
        else:
            cos = dot(written_vec, attraction_vec)/denom
        key = (attraction_data[row][1], attraction_data[row][2])
        if cos > scoretracker[key]:
            scoretracker[key] = cos
            indextracker[key] = row

    # print(indextracker.keys())
    target = []
    # extract the top 3 (can change) and return
    top_n = 3
    
    for key, val in sorted(scoretracker.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        target.append(key)

    outputdata = [attraction_data[indextracker[key]] for key in target]
    keys = ["title", "text"]
    return json.dumps([dict(zip(keys, i)) for i in outputdata])

def attraction_svd(city, written_text):
    print("attraction svd called")
    # Fetch descriptions of attractions from the specified city
    query_sql = f"""SELECT * FROM attractions WHERE City = '{city}'"""
    attraction_data = mysql_engine.query_selector(query_sql)
    attraction_data = attraction_data.all()
    query_sql = f"SELECT * FROM attractions WHERE City = '{city}'"
    attraction_data = mysql_engine.query_selector(query_sql)
    attraction_data = attraction_data.all()
    # Check if data is empty
    if not attraction_data:
        return json.dumps([])
    # Extract descriptions and include the written_text as part of the corpus for vectorization
    descriptions = [row[2] for row in attraction_data] + [written_text]
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    # Perform SVD to reduce the dimensions
    svd_model = TruncatedSVD(n_components=100, random_state=42)  # n_components can be tuned
    reduced_matrix = svd_model.fit_transform(tfidf_matrix)
    # The last row corresponds to the vector for written_text
    written_vec = reduced_matrix[-1]
    attraction_vecs = reduced_matrix[:-1]
    # Compute cosine similarities
    similarities = cosine_similarity([written_vec], attraction_vecs)[0]
    # Sort attractions based on similarity score
    sorted_indices = np.argsort(-similarities)
    top_n = 3
    top_results = [attraction_data[i] for i in sorted_indices[:top_n]]
    # Prepare the output
    keys = ["City", "Location_Name", "Description"]
    return json.dumps([dict(zip(keys, result)) for result in top_results])

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
    # print(jsonify(matched_cities))
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

@app.route("/find_places")
def find_places():
    city = request.args.get('city','')
    rankings = request.args.getlist('rankings')
    prompt = request.args.get('promptDescription','')
    if not city or not rankings or not prompt:
        return jsonify({"error": "Missing required parameters"}), 400
    hotels = hotel_search(city, rankings, None, prompt)
    attractions = attraction_svd(city, prompt) 
    hotels_dict = json.loads(hotels)
    attractions_dict = json.loads(attractions)
    combined_results = {
        "Recommended Hotels": hotels_dict,
        "Recommended Attractions": attractions_dict
    }
    return jsonify(combined_results)

if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
