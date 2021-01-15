package com.reactlibraryocr;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
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
import com.facebook.react.bridge.Promise;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import org.opencv.core.*;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfByte;
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

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

import org.tensorflow.lite.Interpreter;

public class RNOCRModule extends ReactContextBaseJavaModule {

  private final ReactApplicationContext reactContext;
  private Interpreter tfLite;
  private int inputHeight = 1024;
  private int inputWidth = 128;
  private int batchSize = 1;
  private int outSeqLen = 256;
  private int charSetSize = 98;
  private String chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ";

  public RNOCRModule(ReactApplicationContext reactContext) {
    super(reactContext);
    this.reactContext = reactContext;
  }

  @Override
  public String getName() {
    return "OCRManager";
  }

  @ReactMethod
  public void loadModel(
    final String modelPath,
    final int numThreads,
    Promise promise
  ) {
    try{
        AssetManager assetManager = reactContext.getAssets();
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

        final Interpreter.Options tfliteOptions = new Interpreter.Options();
        tfliteOptions.setNumThreads(numThreads);
        tfLite = new Interpreter(buffer, tfliteOptions);

        WritableMap map = Arguments.createMap();
        map.putString("status", "Loading Done");

        promise.resolve(map);
    }
    catch(Exception e){
        promise.reject(e);
    }
  }

  private ByteBuffer convertMattoTfLiteInput(Mat mat)
  {
    ByteBuffer imgData = ByteBuffer.allocateDirect(
        Float.BYTES*batchSize*inputHeight*inputWidth*1
    );
    int pixel = 0;
    for (int i = 0; i < inputHeight; ++i) {
        for (int j = 0; j < inputWidth; ++j) {
            imgData.putFloat((float)mat.get(i,j)[0]);
        }
    }
    return imgData;
  }

  private int[] maxProbIndex(float[][] probs) {
    int[] indices = new int[probs.length];
    int maxIndex = -1;
    float maxProb = 0.0f;

    for(int j = 0; j<probs.length; j++) {
        maxIndex = -1;
        maxProb = 0.0f;
        for (int i = 0; i < probs[j].length; i++){
            if (probs[j][i] > maxProb) {
                maxProb = probs[j][i];
                maxIndex = i;
            }
        }
        indices[j] = maxIndex;
    }
    return indices;
  }

  @ReactMethod
  public void infer(
    final String imageUri,
    Promise promise
  ){
    try{
        Mat frame = Imgcodecs.imread(
            imageUri.replace("file://", ""),
            Imgcodecs.IMREAD_GRAYSCALE
        );
        Size size = new Size(inputHeight, inputWidth);
        Mat img = new Mat();
        Imgproc.resize(frame, img, size);
        ByteBuffer imgData = convertMattoTfLiteInput(img);

        char labels[] = chars.toCharArray();

        float [][][] probsArray = new float[1][outSeqLen][charSetSize];

        if(imgData != null){
            tfLite.run(imgData, probsArray);
        }

        int[] indices = maxProbIndex(probsArray[0]);
        char[] letters = new char[indices.length];
        for(int i = 0; i<indices.length; i++){
            int index = indices[i];
            if(index != 0){
                letters[i] = labels[index - 1];
            }
            else{
                letters[i] = 0;
            }
        }

        String str = String.valueOf(letters);
        WritableMap map = Arguments.createMap();
        map.putString("output", str);
        frame.release();
        img.release();
        promise.resolve(map);
    }
    catch(Exception e){
        promise.reject(e);
    }
  }
}
