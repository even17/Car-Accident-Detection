import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from twilio.rest import Client
import cv2

account_sid = 'AC3ee91700d2d84bf6851054f9d79df7a2'
auth_token = '9f3e68ad69db0ea4a0cfc0b7f3b0d7e5'
client = Client(account_sid, auth_token)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def predict():
    # Replace this with the path to your image
    image = Image.open('static/images/test_image.jpg')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    idx = np.argmax(prediction)
    #print(num)

    if idx == 0:
        
        message = client.messages \
        .create(
                body = "Accident Detected",
                from_='12568139511',
                to="+256705449378"
            )

        print('SMS Sent')
        
        return "Accident Detected"

    elif idx == 1:
        return "No Accident Detected"
    else:
        return "Unknown Vehicle"
		

def get_frame():
	webcam = cv2.VideoCapture(0)
	while True:
		ret, frame = webcam.read()
		if ret == False:
			break
		cv2.imwrite('static/images/test_image.jpg',frame)
		result = predict()
		cv2.putText(frame, result, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))

		#talk(result)                                    #speech output

		imgencode=cv2.imencode('.jpg',frame)[1]
		stringData=imgencode.tostring()

		yield (b'--frame\r\n'
			b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

	webcam.release()
	cv2.destroyAllWindows()