def process_results(doc, results):
    gold_label_set = doc["answers"]
    accuracy = 0.
    prediction = doc["entities"]
    for pred in prediction:
        if pred == gold_label_set:
            accuracy = 1.
            break
    return {
        "accuracy": accuracy,
    } 