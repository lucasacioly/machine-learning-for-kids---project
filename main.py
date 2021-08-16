from mlforkids import MLforKidsImageProject

# treat this key like a password and keep it secret!
key = "736b6b00-fe15-11eb-a100-297677f3c39092f0f662-b0ae-4952-a263-fd81ba11136a"

# this will train your model and might take a little while
myproject = MLforKidsImageProject(key)
myproject.train_model()

# CHANGE THIS to the image file you want to recognize
demo = myproject.prediction("my-test-image.jpg")

label = demo["class_name"]
confidence = demo["confidence"]

# CHANGE THIS to do something different with the result
print("result: '%s' with %d%% confidence" % (label, confidence))
