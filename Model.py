import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import io

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

app = FastAPI()

def preprocess_image(file_bytes):
    file_bytes = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
    return resized.reshape(-1, 40000) / 255.0

@app.post("/verify-signature")
async def verify_signature(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    try:
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()
        resized1 = preprocess_image(img1_bytes)
        resized2 = preprocess_image(img2_bytes)

        out = sess.run(output, feed_dict={
            left_input: resized1,
            right_input: resized2,
            is_training: False,
            prob: 0.1
        })
        out = sess.run(tf.nn.softmax(out))
        label = np.argmax(out)

        if label == 0:
            result = {"match": True, "message": "The signatures match!"}
        else:
            result = {"match": False, "message": "The two signatures do not match!"}
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("Model:app", host="0.0.0.0", port=8000, reload=False)
    # if label == 0:
    #     root = Tk()
    #     root.withdraw()
    #     messagebox.showinfo("Result", "The signatures match!")
    #     root.destroy()
    # else:
    #     root = Tk()
    #     root.withdraw()
    #     messagebox.showwarning("Result", "The two signatures do not match!")
    #     root.destroy()

# except Exception as e:
#     print("Something went wrong! Please try again.", str(e))