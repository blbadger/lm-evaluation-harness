def process_results(doc, results):
    gold_label = doc["answer"].split('#### ')[1]
    accuracy = 0.
    prediction = results
    for pred in prediction[0]:
        if pred.strip(' %$@!*,.') == gold_label:
            accuracy = 1.
            break
    return {
        "accuracy": accuracy,
    } 
