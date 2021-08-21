import os
import pickle
from sklearn.naive_bayes import GaussianNB

# define the class encodings and reverse encodings
classes = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
r_classes = {y: x for x, y in classes.items()}

# function to process data and return it in correct format
def process_data(data):
    processed = [
        {
            "bill_length": d.bill_length,
            "bill_depth": d.bill_depth,
            "flipper_length": d.flipper_length,
            "body_mass": d.body_mass,
            "species": d.species,
        }
        for d in data
    ]

    return processed
