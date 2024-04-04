import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Print keys and inspect data_dict
print(data_dict.keys())
print(data_dict)

# Check shapes of individual elements in the 'data' array
shapes = set()
for element in data_dict['data']:
    shapes.add(np.asarray(element).shape)

if len(shapes) > 1:
    print("Data elements have inconsistent shapes:", shapes)
    # Process your data here to ensure consistency if needed
    # For example, you might pad sequences or resize images

    # Assuming all elements need to be of the same shape
    max_shape = max(shapes)  # Get the maximum shape
    processed_data = []
    for element in data_dict['data']:
        # Process each element to ensure it has the same shape as the maximum shape
        # For example, if element is a list, you can pad it with zeros
        processed_element = np.asarray(element)
        if processed_element.shape != max_shape:
            # Pad the array if needed
            pad_width = [(0, max_dim - curr_dim) for max_dim, curr_dim in zip(max_shape, processed_element.shape)]
            processed_element = np.pad(processed_element, pad_width, mode='constant')
        processed_data.append(processed_element)

    # Convert processed data to NumPy array
    data = np.asarray(processed_data)
else:
    # Convert to NumPy array directly if all elements have the same shape
    data = np.asarray(data_dict['data'])

labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()