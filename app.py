import os
import cv2
import numpy as np
import streamlit as st


class FaceDetectionApp:
    """
    A Streamlit-based application for detecting faces in uploaded images using the
    Haar Cascade classifier.

    Attributes:
        face_cascade (cv2.CascadeClassifier): Pre-trained model for face detection.
        cascade_path (str): Path to the Haar Cascade XML file.
    """

    def __init__(self):
        """
        Initializes the FaceDetectionApp by loading the pre-trained Haar Cascade
        classifier for face detection.
        """
        self.cascade_path = os.path.join(
            "cascade-model", "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

    def detect_faces(self, image):
        """
        Detects faces in the given image and draws rectangles around them.

        Args:
            image (np.ndarray): A NumPy array representing the input image in BGR format.

        Returns:
            np.ndarray: The modified image with rectangles drawn around detected faces.
        """
        # Convert the image to grayscale for face detection
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image using the Haar Cascade classifier
        # Parameters:
        # - scaleFactor: How much the image size is reduced at each scale
        # - minNeighbors: Number of detected neighbors that should be retained
        # - minSize: Minimum face size to consider
        faces = self.face_cascade.detectMultiScale(
            gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return image

    def load_image(self, uploaded_file):
        """
        Loads and processes an uploaded image from Streamlit.

        Args:
            uploaded_file (streamlit.File): The uploaded file object.

        Returns:
            np.ndarray or None: The loaded image as a NumPy array, or None if the file is invalid.
        """
        if uploaded_file is None:
            return None

        try:
            # Read the file as bytes
            img_bytes = uploaded_file.read()
            # Convert to NumPy array
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            # Decode the image using OpenCV
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None

        if img is not None:
            return img
        return None

    def display_image(self, image):
        """
        Displays the processed image in Streamlit.

        Args:
            image (np.ndarray): A NumPy array representing the processed image in BGR format.
        """
        if image is None:
            return

        # Convert BGR to RGB for Streamlit compatibility
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Display the image with full width
        st.image(rgb_image, channels="RGB",
                 caption="Detected Faces", use_container_width=True)

    def run(self):
        """
        Runs the Face Detection App by:
        1. Setting up the Streamlit UI with instructions.
        2. Loading an image from the user.
        3. Detecting faces and displaying the result.
        """
        st.title("Face Detection App")

        # User instructions
        st.markdown(
            """
            ### 📸 Upload an Image for Face Detection

            1. Click the **"Browse files"** button.
            2. Choose a photo or photos in `.jpg`, `.jpeg`, or `.png` format.
            3. The app will detect frontal faces from the photo and display the result in a rectangle.

            You can also drag and drop more than one image at a time into this area.
            """
        )

        # Allow multiple image uploads
        uploaded_files = st.file_uploader(
            "Upload Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="face-detection-uploader"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Process each image
                img = self.load_image(uploaded_file)
                if img is not None:
                    detected_image = self.detect_faces(img.copy())
                    self.display_image(detected_image)
        else:
            st.write("Upload images to see face detection in action!")


# Entry point
if __name__ == "__main__":
    app = FaceDetectionApp()
    app.run()
