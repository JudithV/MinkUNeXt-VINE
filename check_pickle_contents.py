import pickle
import os
from config import PARAMS

file_training = "PNV_training_queries_vmd.pickle"
file_test = "PNV_test_queries_vmd.pickle"

p = os.path.join(PARAMS.dataset_folder, file_training)
with open(p, 'rb') as f:
    database_sets = pickle.load(f)

p = os.path.join(PARAMS.dataset_folder, file_test)
with open(p, 'rb') as f:
    query_sets = pickle.load(f)

print(len(database_sets))
print(len(query_sets))
num_evaluated = 0
for i in range(len(query_sets)):
    for j in range(len(query_sets)):
        if i == j:
            continue
    queries_output = query_sets[i]
    print(queries_output)
    print(f"query_sets[{j}] type: {type(query_sets[j])}")
    print(f"query_sets[{j}].keys(): {list(query_sets[j].keys()) if isinstance(query_sets[j], dict) else 'Not a dict'}")

    for n in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[j][n]    # {'query': path, 'northing': , 'easting': }
        print(query_details)
        true_neighbors = query_details[i]
        if len(true_neighbors) == 0:
            continue
        #print(true_neighbors)
        num_evaluated += 1
print(num_evaluated)
