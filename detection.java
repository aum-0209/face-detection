import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;

/**
 * FaceDetectorApp: Main class to demonstrate face detection using OpenCV.
 * This program loads an image, applies a Haar Cascade classifier, and
 * saves the result with bounding boxes drawn around detected faces.
 * Developed by AUM Vision Technologies.
 */
public class FaceDetectorApp {

    // Configuration constants
    private static final String HAAR_CASCADE_MODEL_PATH = "haarcascade_frontalface_default.xml";
    private static final String INPUT_IMAGE_FILE = "input.jpg";
    private static final String OUTPUT_IMAGE_FILE = "output_faces.jpg";

    public static void main(String[] args) {
        // Load the OpenCV native library
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Error: Native OpenCV library not found. Ensure 'opencv_java" + Core.VERSION + "' is in your path.");
            e.printStackTrace();
            return;
        }

        System.out.println("OpenCV Library Loaded (Version: " + Core.VERSION + ")");

        // Initialize the processor and run the detection pipeline
        ImageProcessor processor = new ImageProcessor(HAAR_CASCADE_MODEL_PATH);
        processor.runDetectionPipeline(INPUT_IMAGE_FILE, OUTPUT_IMAGE_FILE);
    }
}

/**
 * ImageProcessor: Contains the core logic for image loading, face detection,
 * and result manipulation using OpenCV functions.
 */
class ImageProcessor {

    private final CascadeClassifier faceDetector;
    private final Scalar boundingBoxColor = new Scalar(50, 255, 50); // BGR: Light Green
    private static final int THICKNESS = 3;

    /**
     * Initializes the ImageProcessor with the path to the Haar Cascade model.
     * @param cascadeModelPath The path to the XML file defining the classifier.
     */
    public ImageProcessor(String cascadeModelPath) {
        if (!new File(cascadeModelPath).exists()) {
            throw new IllegalArgumentException("Error: Cascade classifier XML file not found at " + cascadeModelPath);
        }
        this.faceDetector = new CascadeClassifier(cascadeModelPath);
    }

    /**
     * Executes the full face detection and image writing process.
     * @param inputPath Path to the source image.
     * @param outputPath Path to save the output image.
     */
    public void runDetectionPipeline(String inputPath, String outputPath) {
        // 1. Load Image
        Mat image = Imgcodecs.imread(inputPath);

        if (image.empty()) {
            System.err.println("Error: Could not load image from " + inputPath + ". File may not exist or is corrupted.");
            return;
        }

        // 2. Detect Faces
        MatOfRect detectedFaces = detectFaces(image);
        Rect[] faceArray = detectedFaces.toArray();
        System.out.println(faceArray.length + " face(s) detected.");

        // 3. Draw Bounding Boxes
        Mat outputImage = drawBoundingBoxes(image, faceArray);

        // 4. Save Result
        boolean success = Imgcodecs.imwrite(outputPath, outputImage);

        if (success) {
            System.out.println("Result saved successfully to: " + outputPath);
        } else {
            System.err.println("Error: Failed to write image to " + outputPath);
        }
    }

    /**
     * Performs face detection on the input image matrix.
     * @param inputImage The source image matrix (Mat).
     * @return A MatOfRect containing the bounding boxes of detected faces.
     */
    private MatOfRect detectFaces(Mat inputImage) {
        MatOfRect faceDetections = new MatOfRect();
        
        // The detectMultiScale method performs the actual detection.
        // The input image is often converted to grayscale and equalized for better results,
        // but for simplicity, we'll use the color image as is here, as the original code did.
        faceDetector.detectMultiScale(inputImage, faceDetections);
        
        return faceDetections;
    }

    /**
     * Draws green bounding boxes on the image for each detected face.
     * @param image The image matrix to draw on.
     * @param faces An array of Rect objects representing face bounding boxes.
     * @return The modified image matrix with bounding boxes.
     */
    private Mat drawBoundingBoxes(Mat image, Rect[] faces) {
        for (Rect r : faces) {
            // Draw a rectangle using the detected coordinates
            Imgproc.rectangle(
                image,
                new Point(r.x, r.y), // Top-left corner
                new Point(r.x + r.width, r.y + r.height), // Bottom-right corner
                boundingBoxColor,
                THICKNESS
            );
        }
        return image;
    }
}
