import numpy
import os
import json
from openai_interface import OpenAI
from sklearn.metrics import f1_score, precision_score, recall_score


def main():
    # Dev set
    dev_path = "training_data/dev.json"
    with open(dev_path) as json_file:
        dev = json.load(json_file)
        
    with open("system_prompt.txt", "r") as sys_prompt:
        system_prompt = sys_prompt.read()

    # Example instance
    print(dev[list(dev.keys())[1]])

    uuid_list = list(dev.keys())
    statements = []
    for i in range(len(uuid_list)):
        # Retrieve all statements from the development set
        statements.append(dev[uuid_list[i]]["Statement"])

    # Predict with OpenAI models
    final_results = {}

    for i in range(len(uuid_list)):
        query = ""

        primary_ctr_path = os.path.join("training_data/CT json", dev[uuid_list[i]]["Primary_id"]+".json")
        with open(primary_ctr_path) as json_file:
            primary_ctr = json.load(json_file)

        # Read section name
        section_name = dev[uuid_list[i]]["Section_id"]

        # Retrieve the full section from the primary trial
        primary_section = primary_ctr[section_name]
        query += f"{section_name}:\n"
        query += "\n".join(primary_section)

        if dev[uuid_list[i]]["Type"] == "Comparison":
            secondary_ctr_path = os.path.join("training_data/CT json", dev[uuid_list[i]]["Secondary_id"] + ".json")
            with open(secondary_ctr_path) as json_file:
                secondary_ctr = json.load(json_file)
            secondary_section = secondary_ctr[dev[uuid_list[i]]["Section_id"]]
            query += "\n".join(secondary_section)

        # Our prediction code here
        predictor = OpenAI()
        prediction = predictor.get_response_message([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ])

        # TODO Check if the prediction is correct, retry if not

        final_results[str(uuid_list[i])] = {"Prediction": prediction}

    # Save the results in the submission format.
    print(final_results)
    with open("results.json",'w') as jsonFile:
        jsonFile.write(json.dumps(final_results, indent=4))

    # Compute F1 score, Precision, and Recall. Note that in the final evaluation systems will be ranked
    # by Faithfulness and Consistency, which cannot be computed on the training and development set.

    gold = dev
    uuid_list = list(final_results.keys())

    results_pred = []
    gold_labels = []
    for i in range(len(uuid_list)):
        if final_results[uuid_list[i]]["Prediction"] == "Entailment":
            results_pred.append(1)
        else:
            results_pred.append(0)
        if gold[uuid_list[i]]["Label"] == "Entailment":
            gold_labels.append(1)
        else:
            gold_labels.append(0)

    f_score = f1_score(gold_labels, results_pred)
    p_score = precision_score(gold_labels, results_pred)
    r_score = recall_score(gold_labels, results_pred)

    print('F1:{:f}'.format(f_score))
    print('precision_score:{:f}'.format(p_score))
    print('recall_score:{:f}'.format(r_score))


if __name__ == "__main__":
    main()
