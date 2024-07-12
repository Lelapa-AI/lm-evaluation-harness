from sklearn.metrics import f1_score


def doc_to_target_hausa(doc):
    replacements = {
        "kimiyya": "Science",
        "fasaha": "Technology",
        "addini": "Religion",
        "kiwon lafiya": "Health",
        "lafiya": "Health",
        "nishaɗi": "Entertainment",
        "Kasuwanci": "Business",
        "siyasa": "Politics",
        "wasanni": "Sports",
        "balaguro": "Travel",
        "afirka": "Africa",
        "duniya": "World",
        "najeriya": "Nigeria",
        "ko wanda ba'a rarraba": "Unclassified"
    }
    return replacements[doc['targets'].lower()]


def doc_to_target_swahili(doc):
    replacements = {
        "biashara": "Business",
        "burudani": "Entertainment",
        "afya": "Health",
        "siasa": "Politics",
        "dini": "Religion",
        "spoti": "Sports",
        "sayansi": "Science",
        "teknolojia": "Technology",
        "usafiri": "Travel",
        "haijaainishwa": "Unclassified"
    }
    return replacements[doc['targets'].lower()]


def doc_to_target_yoruba(doc):
    replacements = {
        "ẹ̀sìn": "Health",
        "eré ìdarayá": "Entertainment",
        "ìdánilárayá": "Sports",
        "ìlera": "Health",
        "òṣèlú": "Politics",
        "ọ̀kọ̀wò": "Business",
        "ìmọ̀-ìjìnlẹ̀": "Science",
        "ìmọ̀-ẹ̀rọ": "Technology",
        "ìrìn-ajò": "Travel",
        "àifisí-ìpele": "Unclassified"
    }
    return replacements[doc['targets'].lower()]


def weighted_f1_score(items):

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="weighted")
    return fscore
