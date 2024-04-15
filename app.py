import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import imutils
from flask import Flask, render_template, request
from io import BytesIO
import base64

app = Flask(__name__, template_folder='templates')

def process_image(input_image):
    clusters = 6  # You can adjust this value

    # Convert to RGB color space
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    input_image = input_image.copy()
    print('Org image shape --> ', input_image.shape)

    img = imutils.resize(input_image, height=200)
    print('After resizing shape --> ', img.shape)

    flat_img = np.reshape(img, (-1, 3))
    print('After Flattening shape --> ', flat_img.shape)

    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(flat_img)

    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint8')

    percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]
    p_and_c = zip(percentages, dominant_colors)
    p_and_c = sorted(p_and_c, reverse=True)

    print("Dominant colors:", p_and_c)  # Print color values for debugging

    output_image = visualize_results(input_image, p_and_c, input_image.shape)

    # Save the processed image
    output_buffer = BytesIO()
    plt.imsave(output_buffer, output_image, format='png', cmap='viridis')  # Specify colormap
    output_buffer.seek(0)

    # Encode the image as base64
    encoded_image = base64.b64encode(output_buffer.read()).decode('utf-8')

    return encoded_image

def visualize_results(input_image, p_and_c, original_shape):
    rows = 1000
    cols = int((input_image.shape[1] / input_image.shape[0]) * rows)
    img = cv2.resize(input_image, dsize=(cols, rows), interpolation=cv2.INTER_LINEAR)

    final = img.copy()
    cv2.rectangle(final, (cols // 2 - 250, rows // 2 - 100), (cols // 2 + 250, rows // 2 + 90), (255, 255, 255), -1)
    cv2.putText(final, 'Most Dominant Colors in the Image', (cols // 2 - 230, rows // 2 - 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    # Add individual color labels
    start = cols // 2 - 220
    for i in range(6):
        end = start + 65  # Reduce the size of rectangles

        # Draw a rectangle around each color label
        cv2.rectangle(final, (start - 5, rows // 2), (end + 5, rows // 2 + 50), p_and_c[i][1].tolist(), -1)
        
        # Add the number in the center of the rectangle
        number_text = str(i + 1)
        text_size = cv2.getTextSize(number_text, cv2.FONT_HERSHEY_DUPLEX, 1, 1)[0]
        text_position = ((start + end - text_size[0]) // 2, rows // 2 + 35)
        cv2.putText(final, number_text, text_position, cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        start = end + 25
    
    return final


@app.route('/', methods=['GET', 'POST'])
def index():
    image_data = ""

    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Read the image file
            img_array = np.frombuffer(uploaded_file.read(), np.uint8)
            input_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Process the image
            image_data = process_image(input_image)

    # Render the HTML template with the processed image data
    return render_template('index.html', image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)
