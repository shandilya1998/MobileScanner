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
  }
}
