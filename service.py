from flask import Flask, jsonify, request, render_template
# import necessary libraries
import pandas as pd
import re
import numpy as np
from nltk import pos_tag
from nltk.tokenize import word_tokenize

import cloudpickle

app = Flask(__name__)

@app.route('/predict-quote', methods = ["GET", "POST"])

def predict_student_interface():
    
    output = None
    if request.method == "POST":


        def sentence_length(row):
            # Using regex to find all words, including those with quotes such as what's
            split_str = re.findall(r"[\w+|\w+\'\w+]+", row['title'])
            row['title_length'] = len(split_str)
            return row

        def word_length(row):
            # Using regex to find all words, including those with quotes such as what's
            split_str = re.findall(r"[\w+|\w+\'\w+]+", row['title'])
            num_words = row['title_length']
            total_len = sum([len(w) for w in split_str])
            row['average_length'] = total_len/num_words
            return row

        def word_tagger(row):
            tagged_list = pos_tag(word_tokenize(row['title']))
            # get all tags
            tags_tuples = [w[1] for w in tagged_list]
            # get unique tags and count
            tags, counts = np.unique(tags_tuples, return_counts=True)
            for i, tag in enumerate(tags):
                row[tag] = counts[i]
            return row

        # check if columns in the list of word tag names, if not add column and give 0 as value
        def empty_col(row):
            word_tag_cols = ['#', '$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW',
            'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT',
            'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB',
            'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', "``"]
            for col in word_tag_cols:
                if col not in row.index:
                    row[col] = 0
            return row

        sentence = str(request.form["sentence"])
        sentence = [sentence]


        model = cloudpickle.load(open('./finalized_model.pkl', 'rb'))

        sentence_df = pd.DataFrame(sentence, columns = ["title"])
        sentence_df = sentence_df.apply(sentence_length, axis = 1)
        sentence_df = sentence_df.apply(word_length, axis = 1)
        sentence_df = sentence_df.apply(word_tagger, axis = 1)
        sentence_df = sentence_df.apply(empty_col, axis = 1)
        output_num = model.predict(sentence_df)[0]
        output = 'Profound' if output_num == 1 else "Not profound"
        #output = sentence_df['title_length'].values[0]

    return render_template("prediction.html", output = output)
