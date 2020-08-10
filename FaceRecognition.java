//Package
package global.skymind.solution.facial_recognition;

//Dependencies  or Libraries for detection and identification
import global.skymind.solution.facial_recognition.detection.FaceDetector;
import global.skymind.solution.facial_recognition.detection.FaceLocalization;
import global.skymind.solution.facial_recognition.detection.OpenCV_DeepLearningFaceDetector;
import global.skymind.solution.facial_recognition.detection.OpenCV_HaarCascadeFaceDetector;
import global.skymind.solution.facial_recognition.identification.DistanceFaceIdentifier;
import global.skymind.solution.facial_recognition.identification.FaceIdentifier;
import global.skymind.solution.facial_recognition.identification.Prediction;
import global.skymind.solution.facial_recognition.identification.feature.RamokFaceNetFeatureProvider;
import global.skymind.solution.facial_recognition.identification.feature.VGG16FeatureProvider;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static global.skymind.solution.facial_recognition.detection.FaceDetector.OPENCV_DL_FACEDETECTOR; //Deep Learning
import static global.skymind.solution.facial_recognition.detection.FaceDetector.OPENCV_HAAR_CASCADE_FACEDETECTOR; //Haar Cascade classifier
import static global.skymind.solution.facial_recognition.identification.FaceIdentifier.FEATURE_DISTANCE_RAMOK_FACENET_PREBUILT; //faceNet Unified Embedding
import static global.skymind.solution.facial_recognition.identification.FaceIdentifier.FEATURE_DISTANCE_VGG16_PREBUILT;
import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;

import java.io.IOException;
import java.util.List;

public class FaceRecognition
{
    private static final Logger log = LoggerFactory.getLogger(FaceRecognitionWebcam.class);
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private static final String outputWindowsName = "CERTIFAI Online CDLE";

    public static void main(String[] args) throws IOException, ClassNotFoundException
    {
        //Face detector & face identifier selection
        //FaceDetector FaceDetector = getFaceDetector(OPENCV_HAAR_CASCADE_FACEDETECTOR);
        FaceDetector FaceDetector = getFaceDetector(OPENCV_DL_FACEDETECTOR);
        FaceIdentifier FaceIdentifier = getFaceIdentifier(FEATURE_DISTANCE_RAMOK_FACENET_PREBUILT);

        //Streaming video frame from camera
        VideoCapture capture = new VideoCapture();
        capture.set(CAP_PROP_FRAME_WIDTH, WIDTH);
        capture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
        namedWindow(outputWindowsName, WINDOW_NORMAL);
        resizeWindow(outputWindowsName, 1280, 720);

        if (!capture.open(0))
        {
            System.out.println("Cannot open the camera!");
        }

        Mat image = new Mat();
        Mat cloneCopy = new Mat();

        while (capture.read(image))
        {
            flip(image, image, 1);

            //Performing face detection
            image.copyTo(cloneCopy);
            FaceDetector.detectFaces(cloneCopy);
            List<FaceLocalization> faceLocalizations = FaceDetector.getFaceLocalization();
            annotateFaces(faceLocalizations, image);

            //Performing face recognition
            image.copyTo(cloneCopy);
            List<List<Prediction>> faceIdentities = FaceIdentifier.recognize(faceLocalizations, cloneCopy);
            labelIndividual(faceIdentities, image);

            //Displaying output in a window
            imshow(outputWindowsName, image);

            char key = (char) waitKey(20);
            // Exit this loop on escape:
            if (key == 27)
            {
                destroyAllWindows();
                break;
            }
        }
    }//End of psvm

    //Getting FaceDetector Class Function
    private static FaceDetector getFaceDetector(String faceDetector) throws IOException
    {
        switch (faceDetector)
        {
            case OPENCV_HAAR_CASCADE_FACEDETECTOR:
                return new OpenCV_HaarCascadeFaceDetector();
            case OPENCV_DL_FACEDETECTOR:
                return new OpenCV_DeepLearningFaceDetector(300, 300, 0.8);
            default:
                return  null;
        }
    }// End of private static FaceDetector getFaceDetector

    //Getting FaceIdentifier Class Function
    private static FaceIdentifier getFaceIdentifier(String faceIdentifier) throws IOException, ClassNotFoundException
    {
        switch (faceIdentifier)
        {
            case FaceIdentifier.FEATURE_DISTANCE_VGG16_PREBUILT:
                return new DistanceFaceIdentifier(
                        new VGG16FeatureProvider(),
                        new ClassPathResource("FaceDB_2").getFile(), 0.5, 6);
            case FaceIdentifier.FEATURE_DISTANCE_RAMOK_FACENET_PREBUILT:
                return new DistanceFaceIdentifier(
                        new RamokFaceNetFeatureProvider(),
                        new ClassPathResource("FaceDB_2").getFile(), 0.5, 6);
            default:
                return null;
        }
    }// End of private static FaceIdentifier getFaceIdentifier

    //Drawing bounding box
    private static void annotateFaces(List<FaceLocalization> faceLocalizations, Mat image)
    {
        for (FaceLocalization i : faceLocalizations)
        {
            rectangle(image,new Rect(new Point((int) i.getLeft_x(),(int) i.getLeft_y()), new Point((int) i.getRight_x(),(int) i.getRight_y())), new Scalar(150, 0, 105, 0),2,8,0);
        }
    }//End of private static void annotateFaces

    //Labelling Predicted Individual's Name Function
    private static void labelIndividual(List<List<Prediction>> faceIdentities, Mat image)
    {
        for (List<Prediction> i: faceIdentities)
        {
            for(int j=0; j<i.size(); j++)
            {
                putText(
                        image,
                        i.get(j).toString(),
                        new Point(
                                (int)i.get(j).getFaceLocalization().getLeft_x() + 2,
                                (int)i.get(j).getFaceLocalization().getLeft_y() - 5
                        ),
                        FONT_HERSHEY_TRIPLEX,
                        0.5,
                        Scalar.WHITE
                );
            }
        }
    }//End of private static void labelIndividual

}
