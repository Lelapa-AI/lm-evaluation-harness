from sklearn.metrics import f1_score


def doc_to_choice(doc, language):
    replacements = {
        'hausa': {
            "Kyakkyawa": "Positive",
            "Korau": "Negative",
            "Tsaka-tsaki": "Neutral"
        },
        'swahili': {
            "Chanya": "Positive",
            "Hasi": "Negative",
            "Wastani": "Neutral"
        },
        'yoruba': {
            "Títọ́": "Positive",
            "Òdì": "Negative",
            "Àarín": "Neutral"
        }
    }
    return replacements[language][doc['targets']]

# def doc_to_target_hausa(doc):
#     replacements = {
#         "Kyakkyawa": "Positive",
#         "Korau": "Negative",
#         "Tsaka-tsaki": "Neutral"
#     }
#     return replacements[doc['targets']]
#
#
# def doc_to_target_swahili(doc):
#     replacements = {
#         "Chanya": "Positive",
#         "Hasi": "Negative",
#         "Wastani": "Neutral"
#     }
#     return replacements[doc['targets']]
#
#
# def doc_to_choice_yoruba(doc):
#     replacements = {
#         "Títọ́": "Positive",
#         "Òdì": "Negative",
#         "Àarín": "Neutral"
#     }
#     return replacements[doc['targets']]


def weighted_f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="weighted")
    return fscore
