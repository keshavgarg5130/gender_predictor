import gradio as gr
from sklearn import tree

def classifier(height_in_cm, weight_in_kg, shoe_size):
    clf = tree.DecisionTreeClassifier()

    # [height, weight, shoe_size]
    X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
         [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

    Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
         'female', 'male', 'male']

    # Train the classifier on our data
    clf = clf.fit(X, Y)

    prediction = clf.predict([[height_in_cm, weight_in_kg, shoe_size]])
    return prediction[0]

# Create a Gradio interface
iface = gr.Interface(fn=classifier, inputs=["number", "number", "number"], outputs="text")
iface.launch()

