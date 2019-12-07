import time
import spacy
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from gensim.models.keyedvectors import KeyedVectors
from nltk import word_tokenize

nlp = spacy.load("en_core_web_lg")
model = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)

def exact_match(sentences, true_values):
    print("Computing Exact Matching...")
    predicted_values = []

    for sentence in sentences:
        if "piece of cake" in sentence:
            predicted_values.append(1)
        else:
            predicted_values.append(0)

    accuracy = accuracy_score(true_values, predicted_values)
    precision = precision_score(true_values, predicted_values)
    recall = recall_score(true_values, predicted_values)
    fscore = f1_score(true_values, predicted_values)
    total_positive_match = sum(predicted_values) / len(sentences)

    print("Accuracy:", accuracy)
    print("Precision:", str(precision))
    print("Recall:", str(recall))
    print("F-Score:", str(fscore))
    print("Total-Positive-Match:", str(total_positive_match))

    print("Done!\n")

    return predicted_values


def lemma_match(sentences, true_values):
    print("Computing Lemma Matching...")
    predicted_values = []
    changed_sentences = []

    for sentence in sentences:
        doc = nlp(sentence)
        cur_sentence = []
        for token in doc:
            cur_sentence.append(token.lemma_)
        cur_sentence = " ".join(cur_sentence)
        changed_sentences.append(cur_sentence)

    for sentence in changed_sentences:
        if "piece of cake" in sentence:
            predicted_values.append(1)
        else:
            predicted_values.append(0)

    accuracy = accuracy_score(true_values, predicted_values)
    precision = precision_score(true_values, predicted_values)
    recall = recall_score(true_values, predicted_values)
    fscore = f1_score(true_values, predicted_values)
    print(predicted_values)
    print(true_values)
    print(sum(predicted_values))
    print(len(sentences))
    total_positive_match = sum(predicted_values) / len(sentences)

    print("Accuracy:", accuracy)
    print("Precision:", str(precision))
    print("Recall:", str(recall))
    print("F-Score:", str(fscore))
    print("Total-Positive-Match:", str(total_positive_match))

    print("Done!\n")

    return predicted_values

def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.vocab]
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return []

def get_sublist(indices, li):
    new_list = []
    for index in list(indices):
        new_list.append(li[index])
    return new_list

def has_idiomatic_expression(sentences, true_values, match_predicted_values):
    kf = KFold(n_splits=4)

    total_accuracy1 = 0
    total_precision1 = 0
    total_recall1 = 0
    total_fscore1 = 0
    total_total_positive_match1 = 0

    total_accuracy2 = 0
    total_precision2 = 0
    total_recall2 = 0
    total_fscore2 = 0
    total_total_positive_match2 = 0

    total_accuracy3 = 0
    total_precision3 = 0
    total_recall3 = 0
    total_fscore3 = 0
    total_total_positive_match3 = 0

    total_accuracy4 = 0
    total_precision4 = 0
    total_recall4 = 0
    total_fscore4 = 0
    total_total_positive_match4 = 0

    total_accuracy5 = 0
    total_precision5 = 0
    total_recall5 = 0
    total_fscore5 = 0
    total_total_positive_match5 = 0

    total_accuracy6 = 0
    total_precision6 = 0
    total_recall6 = 0
    total_fscore6 = 0
    total_total_positive_match6 = 0

    for train_index, test_index in kf.split(sentences):
        train_sentences = get_sublist(train_index, sentences)
        test_sentences = get_sublist(test_index, sentences)
        train_true = get_sublist(train_index, true_values)
        test_true = get_sublist(test_index, true_values)
        train_match = get_sublist(train_index, match_predicted_values)
        test_match = get_sublist(test_index, match_predicted_values)

        word2vec_features_train = []

        for i, sentence in enumerate(train_sentences):
            tokens = word_tokenize(sentence)
            mean_vector = get_mean_vector(model, tokens)[:40]
            list_features = list(mean_vector)
            list_features.append(train_match[i])
            word2vec_features_train.append(list_features)

        word2vec_features_test = []

        for i, sentence in enumerate(test_sentences):
            tokens = word_tokenize(sentence)
            mean_vector = get_mean_vector(model, tokens)[:40]
            list_features = list(mean_vector)
            list_features.append(test_match[i])
            word2vec_features_test.append(list_features)


        mlp1 = MLPClassifier(hidden_layer_sizes=(45,), max_iter=2000)
        mlp2 = MLPClassifier(hidden_layer_sizes=(40,), max_iter=2000)
        mlp3 = MLPClassifier(hidden_layer_sizes=(35,), max_iter=2000)
        mlp4 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000)
        mlp5 = MLPClassifier(hidden_layer_sizes=(55,), max_iter=2000)
        mlp6 = MLPClassifier(hidden_layer_sizes=(60,), max_iter=2000)

        mlp1.fit(word2vec_features_train, train_true)
        mlp2.fit(word2vec_features_train, train_true)
        mlp3.fit(word2vec_features_train, train_true)
        mlp4.fit(word2vec_features_train, train_true)
        mlp5.fit(word2vec_features_train, train_true)
        mlp6.fit(word2vec_features_train, train_true)

        predictions1 = list(mlp1.predict(word2vec_features_test))
        predictions2 = list(mlp2.predict(word2vec_features_test))
        predictions3 = list(mlp3.predict(word2vec_features_test))
        predictions4 = list(mlp4.predict(word2vec_features_test))
        predictions5 = list(mlp5.predict(word2vec_features_test))
        predictions6 = list(mlp6.predict(word2vec_features_test))

        accuracy1 = accuracy_score(test_true, predictions1)
        precision1 = precision_score(test_true, predictions1)
        recall1 = recall_score(test_true, predictions1)
        fscore1 = f1_score(test_true, predictions1)
        total_positive_match1 = sum(predictions1) / len(test_sentences)

        accuracy2 = accuracy_score(test_true, predictions2)
        precision2 = precision_score(test_true, predictions2)
        recall2 = recall_score(test_true, predictions2)
        fscore2 = f1_score(test_true, predictions2)
        total_positive_match2 = sum(predictions2) / len(test_sentences)

        accuracy3 = accuracy_score(test_true, predictions3)
        precision3 = precision_score(test_true, predictions3)
        recall3 = recall_score(test_true, predictions3)
        fscore3 = f1_score(test_true, predictions3)
        total_positive_match3 = sum(predictions3) / len(test_sentences)

        accuracy4 = accuracy_score(test_true, predictions4)
        precision4 = precision_score(test_true, predictions4)
        recall4 = recall_score(test_true, predictions4)
        fscore4 = f1_score(test_true, predictions4)
        total_positive_match4 = sum(predictions4) / len(test_sentences)

        accuracy5 = accuracy_score(test_true, predictions5)
        precision5 = precision_score(test_true, predictions5)
        recall5 = recall_score(test_true, predictions5)
        fscore5 = f1_score(test_true, predictions5)
        total_positive_match5 = sum(predictions5) / len(test_sentences)

        accuracy6 = accuracy_score(test_true, predictions6)
        precision6 = precision_score(test_true, predictions6)
        recall6 = recall_score(test_true, predictions6)
        fscore6 = f1_score(test_true, predictions6)
        total_positive_match6 = sum(predictions6) / len(test_sentences)
        
        total_accuracy1 += accuracy1
        total_precision1 += precision1
        total_recall1 += recall1
        total_fscore1+= fscore1
        total_total_positive_match1 += total_positive_match1

        total_accuracy2 += accuracy2
        total_precision2 += precision2
        total_recall2 += recall2
        total_fscore2+= fscore2
        total_total_positive_match2 += total_positive_match2

        total_accuracy3 += accuracy3
        total_precision3 += precision3
        total_recall3 += recall3
        total_fscore3+= fscore3
        total_total_positive_match3 += total_positive_match3

        total_accuracy4 += accuracy4
        total_precision4 += precision4
        total_recall4 += recall4
        total_fscore4+= fscore4
        total_total_positive_match4 += total_positive_match4

        total_accuracy5 += accuracy5
        total_precision5 += precision5
        total_recall5 += recall5
        total_fscore5+= fscore5
        total_total_positive_match5 += total_positive_match5

        total_accuracy6 += accuracy6
        total_precision6 += precision6
        total_recall6 += recall6
        total_fscore6+= fscore6
        total_total_positive_match6 += total_positive_match6

    print("\nAverage scores:")
    print("Accuracy:", total_accuracy1 / 4)
    print("Precision:", total_precision1 / 4)
    print("Recall:", total_recall1 / 4)
    print("F-Score:", total_fscore1 / 4)
    print("Total-Positive-Match:", total_total_positive_match1 / 4, "\n")

    print("Accuracy:", total_accuracy2 / 4)
    print("Precision:", total_precision2 / 4)
    print("Recall:", total_recall2 / 4)
    print("F-Score:", total_fscore2 / 4)
    print("Total-Positive-Match:", total_total_positive_match2 / 4, "\n")

    print("Accuracy:", total_accuracy3 / 4)
    print("Precision:", total_precision3 / 4)
    print("Recall:", total_recall3 / 4)
    print("F-Score:", total_fscore3 / 4)
    print("Total-Positive-Match:", total_total_positive_match3 / 4, "\n")

    print("Accuracy:", total_accuracy4 / 4)
    print("Precision:", total_precision4 / 4)
    print("Recall:", total_recall4 / 4)
    print("F-Score:", total_fscore4 / 4)
    print("Total-Positive-Match:", total_total_positive_match4 / 4, "\n")

    print("Accuracy:", total_accuracy5 / 4)
    print("Precision:", total_precision5 / 4)
    print("Recall:", total_recall5 / 4)
    print("F-Score:", total_fscore5 / 4)
    print("Total-Positive-Match:", total_total_positive_match5 / 4, "\n")

    print("Accuracy:", total_accuracy6 / 4)
    print("Precision:", total_precision6 / 4)
    print("Recall:", total_recall6 / 4)
    print("F-Score:", total_fscore6 / 4)
    print("Total-Positive-Match:", total_total_positive_match6 / 4)

def find_idioms():
    start_time = time.time()

    sentences = []
    true_values = []

    with open("./data/piece_of_cake_training.tsv", encoding="utf8") as fp:
        line = fp.readline()
        while line:
            data = line.lower().split("\t")
            sentences.append(data[0])
            true_values.append(int(data[1]))
            line = fp.readline()
        fp.close()

    #train_sentences, test_sentences, train_true, test_true = train_test_split(sentences, true_values, test_size = 0.3)

    prediction_exact = exact_match(sentences, true_values)
    prediction_lemma = lemma_match(sentences, true_values)
    has_idiomatic_expression(sentences, true_values, prediction_lemma)


    print("--- %s seconds ---" % (round(time.time() - start_time, 2)))


if __name__ == "__main__":
    print("Starting...")

    find_idioms()
