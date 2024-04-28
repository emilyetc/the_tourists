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
from nltk.corpus import wordnet as wn
import re
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
LOCAL_MYSQL_DATABASE = "globe_trotter"
mysql_engine = MySQLDatabaseHandler(
    LOCAL_MYSQL_USER, LOCAL_MYSQL_USER_PASSWORD, LOCAL_MYSQL_PORT, LOCAL_MYSQL_DATABASE
)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
# mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
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
    for term in filtered:
        synsets = wn.synsets(term)
        for synset in synsets:
            for lemma in synset.lemmas():
                synonym = lemma.name()
                if synonym != term:
                    output[synonym.lower()] += output[term] / 2 
    for term, freq in output.items():
        output[term] = freq if freq >= 0 else 0
    return output
def remove_keys_values(dictionary, keys_to_remove):
    for key in keys_to_remove:
        dictionary.pop(key, None)
    return dictionary
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

def hotel_search(city, rankinglst, written_text):
    """city = target city, rankinglst = a string list of user's preferences with index = 0
    being the most important, written_text =
    user's written input
    """
    rankinglst = lstparser(rankinglst)
    # formatting user input
    written_dict = process_text(written_text)
    written_vec = [written_dict[val] for val in written_dict]
    written_dict = remove_keys_values(written_dict, ['hotel', 'hotels', 'place', 'like', 'love', 'attractions', 'room', 'rooms', 'not', 'places', 'stay','just', 'want'])
    written_vec = rocchio(written_dict, good_hotel_reviews, bad_hotel_reviews)

    # selecting hotels within a city 
    query_sql = f"""SELECT * FROM reviews WHERE locality = '{city}'"""
    review_data = mysql_engine.query_selector(query_sql)
    review_data = review_data.all()
    review_data = [[str(col) for col in row] for row in review_data]
    # selecting rankings wtihin a city
    query_sql = f"""SELECT * FROM rankings WHERE locality = '{city}'"""
    ranking_data = mysql_engine.query_selector(query_sql)
    ranking_data = ranking_data.all()
    # tracks the highest score so far
    scoretracker = defaultdict(int)
    # tracks the index of the highest score so far
    indextracker = dict()
    rankingtracker = dict()
    # creating dict to index into rankings column easily
    rankingsindex = {'service': 0, 'cleanliness': 1, 'value': 2, 'location': 3, 'sleep quality': 4, 'rooms': 5}
    ranking_scores = defaultdict(dict)
    starsIndex = {}
    # calculate rankings score
    for row in range(len(ranking_data)):
        # calculate score
        score = 0
        key = (ranking_data[row][0], ranking_data[row][1])
        for val in range(len(rankinglst)):
            score += (6 - val) * ranking_data[row][2 + rankingsindex[rankinglst[val]]]
            ranking_scores[key][list(rankingsindex.keys())[val]] = ranking_data[row][2 + rankingsindex[rankinglst[val]]]
        score /= 105
        starsIndex[key] = ranking_data[row][8]
        rankingtracker[key] = score
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
        curr_score = (cos + rankingtracker[key])/(float(2))
        if curr_score > scoretracker[key]: #cosine sim seems irrelevant, so adding a multiplier
            scoretracker[key] = curr_score
            indextracker[key] = row
    target = []
    # extract the top 3 (can change) and return
    top_n = 10
    '''for key, val in sorted(scoretracker.items(), key=lambda x: x[0], reverse=True)[:top_n]:
        target.append(key)'''
    target = sorted(scoretracker, key=scoretracker.get, reverse=True)[:top_n]
    outputdata = [
    review_data[indextracker[key]] + 
    [str(scoretracker[key]*100)[:5]] + 
    [str(ranking_scores[key][ranking]) for ranking in rankinglst] + [str(starsIndex[key])]
    for key in target
]
    keys = ["ratings", "title", "text", "score"] + rankinglst + ["stars"]
    return json.dumps([dict(zip(keys, i)) for i in outputdata])


def highlight_words(text, words):
    pattern = r'\b(' + '|'.join(re.escape(word) for word in words) + r')\b'
    highlighted = re.sub(pattern, r'<b>\1</b>', text, flags=re.IGNORECASE)
    return highlighted

def attraction_svd2(city, written_text):
    # Modify city names if needed
    if city == 'New York City':
        city = 'New York'
    elif city == 'Washington DC':
        city = 'Washington District of Columbia'

    # Fetch descriptions of attractions from the specified city
    query_sql = f"""SELECT * FROM attractions WHERE City = '{city}' AND Description NOT LIKE '%%hotel%%' AND Description NOT LIKE '%%INN%%'"""
    attraction_data = mysql_engine.query_selector(query_sql)
    attraction_data = attraction_data.all()
    
    # Check if data is empty
    if not attraction_data:
        return json.dumps([])

    # Preprocess input text and attraction descriptions
    written_dict = process_text(written_text)
    processed_descriptions = []

    # Process each attraction's description with process_review
    for city, location_name, description in attraction_data:
        _, modified_description = process_review(description, list(written_dict.keys()))
        processed_descriptions.append(modified_description)
    
    # Include the written_text in the corpus for vectorization
    descriptions = processed_descriptions + [written_text]

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    # Perform SVD to reduce the dimensions
    svd_model = TruncatedSVD(n_components=50) 
    reduced_matrix = svd_model.fit_transform(tfidf_matrix)
    
    # Separate the vector for written_text
    written_vec = reduced_matrix[-1]
    attraction_vecs = reduced_matrix[:-1]
    
    # Compute cosine similarities
    similarities = cosine_similarity([written_vec], attraction_vecs)[0]
    
    # Sort attractions based on similarity score
    sorted_indices = np.argsort(-similarities)
    top_n = 5  # Adjust as needed
    top_results = [attraction_data[i] for i in sorted_indices[:top_n]]
    highlighted_results = []
    relevant_words = remove_keys_values(written_dict,['hotel', 'hotels', 'place', 'like', 'love', 'attractions', 'room', 'rooms', 'not']
)
    relevant_words = list(written_dict.keys())
    for i in range(len(top_results)):
        city, location_name, description = top_results[i]
        highlighted_description = highlight_words(description, relevant_words)
        highlighted_results.append((city, location_name, highlighted_description))

    # Prepare the output
    keys = ["City", "Location_Name", "Description"]
    return json.dumps([dict(zip(keys, result)) for result in highlighted_results])
    # keys = ["City", "Location_Name", "Description"]
    # return json.dumps([dict(zip(keys, result)) for result in top_results])

good_hotel_reviews = set()
bad_hotel_reviews = set()
good_hotel_names = set()
bad_hotel_names = set()

def rocchio(query_dict, good_reviews, bad_reviews, a=0.4, b=0.2, c=0.4):
    query_vector = np.array([query_dict[val] for val in query_dict])
    sum_good_vector = np.zeros(len(query_dict))
    sum_bad_vector = np.zeros(len(query_dict))

    if len(good_reviews) == 0 and len(bad_reviews) == 0:
        return query_vector

    for rev in good_reviews:
      review_dict, _ = process_review(rev, list(query_dict.keys()))
      review_vec = [review_dict[val] for val in query_dict]
      sum_good_vector = np.add(sum_good_vector, review_vec)

    for rev in bad_reviews:
      review_dict, _ = process_review(rev, list(query_dict.keys()))
      review_vec = [review_dict[val] for val in query_dict]
      sum_bad_vector = np.add(sum_bad_vector, review_vec)

    if(len(good_reviews) == 0):
      div2 = 0
    else:
      div2 = (1/len(good_reviews))
    if(len(bad_reviews) == 0):
      div3 = 0
    else:
      div3 = (1/len(bad_reviews))

    q_1 = (a * query_vector) + (b * div2 * sum_good_vector) - (c * div3 * sum_bad_vector)
    
    for i in range(len(q_1)):
      if(q_1[i] < 0):
        q_1[i] = 0
    return q_1

def reset_feedback():
    global good_hotel_reviews
    global bad_hotel_reviews
    global good_hotel_names
    global bad_hotel_names
    good_hotel_reviews = set()
    bad_hotel_reviews = set()
    good_hotel_names = set()
    bad_hotel_names = set()

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

@app.route("/find_places")
def find_places():
    reset_feedback()
    city = request.args.get('city','')
    rankings = request.args.getlist('rankings')
    prompt = request.args.get('promptDescriptionHotel','')
    attractprompt = request.args.get('promptDescriptionAttraction','')
    if not city or not rankings or not prompt or not attractprompt:
        return jsonify({"error": "Missing required parameters"}), 400
    hotels = hotel_search(city, rankings, prompt)
    attractions = attraction_svd2(city, attractprompt) 
    hotels_dict = json.loads(hotels)
    attractions_dict = json.loads(attractions)
    combined_results = {
        "Recommended Hotels": hotels_dict,
        "Recommended Attractions": attractions_dict
    }
    return jsonify(combined_results)

@app.route("/refine_search")
def refine_search():
    city = request.args.get('city','')
    rankings = request.args.getlist('rankings')
    prompt = request.args.get('promptDescription','')
    if not city or not rankings or not prompt:
        return jsonify({"error": "Missing required parameters"}), 400
    hotels = hotel_search(city, rankings, prompt)
    attractions = attraction_svd2(city, prompt) 
    hotels_dict = json.loads(hotels)
    attractions_dict = json.loads(attractions)
    combined_results = {
        "Recommended Hotels": hotels_dict,
        "Recommended Attractions": attractions_dict,
        "Good_Hotel_Names": list(good_hotel_names),
        "Bad_Hotel_Names": list(bad_hotel_names)
    }
    return jsonify(combined_results)

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    data = request.get_json()
    hotel_review = data.get('hotelReview')
    hotel_name = data.get('hotelName')
    button_type = data.get('buttonType')

    if button_type == 'thumbsUp':
        if hotel_review in good_hotel_reviews:
            good_hotel_reviews.remove(hotel_review)
            good_hotel_names.remove(hotel_name)
        else:
            good_hotel_reviews.add(hotel_review)
            good_hotel_names.add(hotel_name)
    elif button_type == 'thumbsDown':
        if hotel_review in bad_hotel_reviews:
            bad_hotel_reviews.remove(hotel_review)
            bad_hotel_names.remove(hotel_name)
        else:
            bad_hotel_reviews.add(hotel_review)
            bad_hotel_names.add(hotel_name)

    return jsonify({"message": "Feedback received successfully"})

if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
