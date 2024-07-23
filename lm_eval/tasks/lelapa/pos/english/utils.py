from sklearn.metrics import f1_score


def doc_to_target(doc):
    return transform_text(doc['targets'])


def transform_text(text):
    pos_pairs = []

    for pair in text.split('\n'):
        if pair.strip():
            word, tag = pair.split(': ')
            word = word.lower()
            tag = tag.upper()
            pos_pairs.append(f"{tag}: {word}")

    # Join all the transformed output lines with $$ as separator
    return ' $$ '.join(pos_pairs)


def weighted_f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="weighted")
    return fscore
