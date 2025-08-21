
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np
import os
from tkinter import messagebox, filedialog, Tk

input_image_size = 200

print("****************Loading the Siamese model****************\n")

# Reset the graph and start session
tf.reset_default_graph()
sess = tf.Session()

# Load the trained model
saver = tf.train.import_meta_graph('./model_params/model.meta')
saver.restore(sess, './model_params/model')
graph = tf.get_default_graph()

# Fetching tensors by name
left_input = graph.get_tensor_by_name("left_input:0")
right_input = graph.get_tensor_by_name("right_input:0")
y = graph.get_tensor_by_name('Y:0')
output = graph.get_tensor_by_name("output:0")
accuracy = graph.get_operation_by_name("accuracy").outputs[0]
is_training = graph.get_tensor_by_name("is_training:0")
prob = graph.get_tensor_by_name("prob:0")

print("\t\tSuccessfully Loaded The Model\t\t")

# Info pop-up
root = Tk()
root.withdraw()
messagebox.showinfo(
    "Information",
    "The program will prompt you to select two signature images that are to be compared and will display if they match or not."
)
root.destroy()

# File selection dialog
root = Tk()
root.withdraw()
answer1 = filedialog.askopenfilename(
    parent=root,
    initialdir=os.getcwd(),
    title="Please select a genuine signature image file"
)
answer2 = filedialog.askopenfilename(
    parent=root,
    initialdir=os.getcwd(),
    title="Please select a signature image that is to be tested"
)
root.destroy()

# Signature Verification Logic
try:
    sig1_img = cv2.imread(answer1, 0)
    resized1 = cv2.resize(sig1_img, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC).reshape(-1, 40000) / 255.0

    sig2_img = cv2.imread(answer2, 0)
    resized2 = cv2.resize(sig2_img, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC).reshape(-1, 40000) / 255.0

    out = sess.run(output, feed_dict={
        left_input: resized1,
        right_input: resized2,
        is_training: False,
        prob: 0.1
    })
    out = sess.run(tf.nn.softmax(out))
    label = np.argmax(out)

    if label == 0:
        root = Tk()
        root.withdraw()
        messagebox.showinfo("Result", "The signatures match!")
        root.destroy()
    else:
        root = Tk()
        root.withdraw()
        messagebox.showwarning("Result", "The two signatures do not match!")
        root.destroy()

except Exception as e:
    print("Something went wrong! Please try again.", str(e))
