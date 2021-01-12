package com.reactlibraryocr;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.provider.MediaStore;
import android.util.Base64;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.Callback;
import com.facebook.react.bridge.WritableMap;

import java.io.FileInputStream;
import java.io.IOException;
import org.opencv.core.*;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfByte;
import org.opencv.dnn.*;
import org.opencv.dnn.Dnn;
import org.opencv.utils.*;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.imgcodecs.*;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import org.opencv.calib3d.Calib3d;



import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

public class RNOCRModule extends ReactContextBaseJavaModule {

  private final ReactApplicationContext reactContext;

  public RNOCRModule(ReactApplicationContext reactContext) {
    super(reactContext);
    this.reactContext = reactContext;
  }

  @Override
  public String getName() {
    return "OCRManager";
  }

  @ReactMethod
  public void detectAndBound(String imageUri, Callback callback){

    Mat frame = Imgcodecs.imread(
      imageUri.replace("file://", ""), 
      Imgproc.COLOR_BGR2RGB
    );
    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB);

    float scoreThresh = 0.5f;
    float nmsThresh = 0.4f;
    Net net = Dnn.readNetFromTensorflow("c:/data/mdl/frozen_east_text_detection.pb");
    // input image

    Size siz = new Size(320, 320);
    int W = (int)(siz.width / 4); // width of the output geometry  / score maps
    int H = (int)(siz.height / 4); // height of those. the geometry has 4, vertically stacked maps, the score one 1
    Mat blob = Dnn.blobFromImage(frame, 1.0,siz, new Scalar(123.68, 116.78, 103.94), true, false);
    net.setInput(blob);
    List<Mat> outs = new ArrayList<>(2);
    List<String> outNames = new ArrayList<String>();
    outNames.add("feature_fusion/Conv_7/Sigmoid");
    outNames.add("feature_fusion/concat_3");
    net.forward(outs, outNames);

    // Decode predicted bounding boxes.
    Mat scores = outs.get(0).reshape(1, H);
    // My lord and savior : http://answers.opencv.org/question/175676/javaandroid-access-4-dim-mat-planes/
    Mat geometry = outs.get(1).reshape(1, 5 * H); // don't hardcode it !
    List<Float> confidencesList = new ArrayList<>();
    List<RotatedRect> boxesList = decode(scores, geometry, confidencesList, scoreThresh);

    // Apply non-maximum suppression procedure.
    MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confidencesList));
    RotatedRect[] boxesArray = boxesList.toArray(new RotatedRect[0]);
    MatOfRotatedRect boxes = new MatOfRotatedRect(boxesArray);
    MatOfInt indices = new MatOfInt();
    Dnn.NMSBoxesRotated(boxes, confidences, scoreThresh, nmsThresh, indices);

    // Render detections
    Point ratio = new Point((float)frame.cols()/siz.width, (float)frame.rows()/siz.height);
    int[] indexes = indices.toArray();
    for(int i = 0; i<indexes.length;++i) {
        RotatedRect rot = boxesArray[indexes[i]];
        Point[] vertices = new Point[4];
        rot.points(vertices);
        for (int j = 0; j < 4; ++j) {
            vertices[j].x *= ratio.x;
            vertices[j].y *= ratio.y;
        }
        for (int j = 0; j < 4; ++j) {
            Imgproc.line(frame, vertices[j], vertices[(j + 1) % 4], new Scalar(0, 0,255), 1);
        }
    }

    Bitmap bitmap = Bitmap.createBitmap(
      frame.cols(), 
      frame.rows(), 
      Bitmap.Config.ARGB_8888
    );
    Utils.matToBitmap(frame, bitmap);

    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    bitmap.compress(Bitmap.CompressFormat.JPEG, 70, byteArrayOutputStream);
    byte[] byteArray = byteArrayOutputStream.toByteArray();

    WritableMap map = Arguments.createMap();
    map.putString("image", Base64.encodeToString(byteArray, Base64.DEFAULT));
    callback.invoke(null, map);
  }

  private static List<RotatedRect> decode(Mat scores, Mat geometry, List<Float> confidences, float scoreThresh) {
    // size of 1 geometry plane
    int W = geometry.cols();
    int H = geometry.rows() / 5;
    //System.out.println(geometry);
    //System.out.println(scores);

    List<RotatedRect> detections = new ArrayList<>();
    for (int y = 0; y < H; ++y) {
        Mat scoresData = scores.row(y);
        Mat x0Data = geometry.submat(0, H, 0, W).row(y);
        Mat x1Data = geometry.submat(H, 2 * H, 0, W).row(y);
        Mat x2Data = geometry.submat(2 * H, 3 * H, 0, W).row(y);
        Mat x3Data = geometry.submat(3 * H, 4 * H, 0, W).row(y);
        Mat anglesData = geometry.submat(4 * H, 5 * H, 0, W).row(y);

        for (int x = 0; x < W; ++x) {
            double score = scoresData.get(0, x)[0];
            if (score >= scoreThresh) {
                double offsetX = x * 4.0;
                double offsetY = y * 4.0;
                double angle = anglesData.get(0, x)[0];
                double cosA = Math.cos(angle);
                double sinA = Math.sin(angle);
                double x0 = x0Data.get(0, x)[0];
                double x1 = x1Data.get(0, x)[0];
                double x2 = x2Data.get(0, x)[0];
                double x3 = x3Data.get(0, x)[0];
                double h = x0 + x2;
                double w = x1 + x3;
                Point offset = new Point(offsetX + cosA * x1 + sinA * x2, offsetY - sinA * x1 + cosA * x2);
                Point p1 = new Point(-1 * sinA * h + offset.x, -1 * cosA * h + offset.y);
                Point p3 = new Point(-1 * cosA * w + offset.x,      sinA * w + offset.y); // original trouble here !
                RotatedRect r = new RotatedRect(new Point(0.5 * (p1.x + p3.x), 0.5 * (p1.y + p3.y)), new Size(w, h), -1 * angle * 180 / Math.PI);
                detections.add(r);
                confidences.add((float) score);
            }
        }
    }
    return detections;
  }
}
