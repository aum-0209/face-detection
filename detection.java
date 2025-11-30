import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;

public class FaceDetectorApp {

    private static final String HAAR_CASCADE_MODEL_PATH = "haarcascade_frontalface_default.xml";
    private static final String INPUT_IMAGE_FILE = "input.jpg";
    private static final String OUTPUT_IMAGE_FILE = "output_faces.jpg";

    public static void main(String[] args) {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Error: Native OpenCV library not found. Ensure 'opencv_java" + Core.VERSION + "' is in your path.");
            e.printStackTrace();
            return;
        }

        ImageProcessor processor = new ImageProcessor(HAAR_CASCADE_MODEL_PATH);
        processor.runDetectionPipeline(INPUT_IMAGE_FILE, OUTPUT_IMAGE_FILE);
    }
}

class ImageProcessor {

    private final CascadeClassifier faceDetector;
    private final Scalar boundingBoxColor = new Scalar(50, 255, 50);
    private static final int THICKNESS = 3;

    public ImageProcessor(String cascadeModelPath) {
        if (!new File(cascadeModelPath).exists()) {
            throw new IllegalArgumentException("Error: Cascade classifier XML file not found at " + cascadeModelPath);
        }
        this.faceDetector = new CascadeClassifier(cascadeModelPath);
    }

    public void runDetectionPipeline(String inputPath, String outputPath) {
        Mat image = Imgcodecs.imread(inputPath);

        if (image.empty()) {
            System.err.println("Error: Could not load image from " + inputPath + ". File may not exist or is corrupted.");
            return;
        }

        MatOfRect detectedFaces = detectFaces(image);
        Rect[] faceArray = detectedFaces.toArray();
        System.out.println(faceArray.length + " face(s) detected.");

        Mat outputImage = drawBoundingBoxes(image, faceArray);

        boolean success = Imgcodecs.imwrite(outputPath, outputImage);

        if (success) {
            System.out.println("Result saved successfully to: " + outputPath);
        } else {
            System.err.println("Error: Failed to write image to " + outputPath);
        }
    }

    private MatOfRect detectFaces(Mat inputImage) {
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(inputImage, faceDetections);
        return faceDetections;
    }

    private Mat drawBoundingBoxes(Mat image, Rect[] faces) {
        for (Rect r : faces) {
            Imgproc.rectangle(
                image,
                new Point(r.x, r.y),
                new Point(r.x + r.width, r.y + r.height),
                boundingBoxColor,
                THICKNESS
            );
        }
        return image;
    }
}
