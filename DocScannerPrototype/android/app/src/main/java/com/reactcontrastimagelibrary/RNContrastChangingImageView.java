package com.reactcontrastimagelibrary;

import android.content.Context;
import android.content.ContentResolver;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.ParcelFileDescriptor;
import android.net.Uri;
import android.util.Log;
import android.app.Activity;

import androidx.appcompat.widget.AppCompatImageView;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.io.InputStream;
import java.io.FileDescriptor;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.MalformedURLException;
import java.util.concurrent.ExecutionException;
import java.util.UUID;

import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.core.DeviceEventManagerModule;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.ReactContext;
import com.facebook.react.uimanager.events.RCTEventEmitter;

public class RNContrastChangingImageView extends AppCompatImageView {
    public static final String TAG = "ContrastEditor";
    private String cacheFolderName = "RNContrastChangingImage";
    private Bitmap initialData = null;
    private Bitmap imageData = null;
    private String imageUri = null;
    private double contrast = 1;
    protected Context mContext;
    public static RNContrastChangingImageView instance = null;
    protected Activity mActivity;
    public String fileName = null;

    public RNContrastChangingImageView(Context context, Activity activity) {
        super(context);
        this.mContext = context;
        this.mActivity = activity;
        //createInstance(context, activity);
    }

    public static RNContrastChangingImageView getInstance() {
        return instance;
    }

    public static void createInstance(Context context, Activity activity) {
        instance = new RNContrastChangingImageView(context, activity);
        
    }

    public void setImageUri(String imgUri) {
        Log.d(TAG, "set image");
        Log.d(TAG, "image source : " + imgUri);
        if (imgUri != this.imageUri) {
            this.imageUri = imgUri;
            try{
                File imgFile = new File(imgUri);
                Bitmap bitmap = BitmapFactory.decodeStream(
                    new FileInputStream(imgFile)
                );
                Log.d(TAG, "set image source");
                this.imageData = bitmap;
                this.initialData = bitmap;
                this.setImageBitmap(bitmap);
            } catch(FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }

    public void setContrast(double contrastVal) {
        this.contrast = contrastVal;

        if (this.imageData != null) {
            this.updateImageContrast();
        }
    }

    public void setResizeMode(String mode) {
        switch (mode) {
            case "cover":
                this.setScaleType(ScaleType.CENTER_CROP);
                break;
            case "stretch":
                this.setScaleType(ScaleType.FIT_XY);
                break;
            case "contain":
            default:
                this.setScaleType(ScaleType.FIT_CENTER);
                break;
        }
    }

    private String generateStoredFileName() throws Exception {
        String folderDir = this.mContext.getCacheDir().toString();
        File folder = new File( folderDir + "/" + this.cacheFolderName);
        if (!folder.exists()) {
            boolean result = folder.mkdirs();
            if (result) {
                Log.d(TAG, "wrote: created folder " + folder.getPath());
            } else {
                Log.d(TAG, "Not possible to create folder");
                throw new Exception("Failed to create the cache directory");
            }   
        }   
        return folderDir + "/" + this.cacheFolderName + "/" + "contrast_editted" + UUID.randomUUID() + ".png"; 
    } 

    private void updateImageContrast() {
        try {
            Mat matImage = new Mat();
            Utils.bitmapToMat(this.initialData, matImage);

            Scalar imgScalVec = Core.sumElems(matImage);
            double[] imgAvgVec = imgScalVec.val;
            for (int i = 0; i < imgAvgVec.length; i++) {
                imgAvgVec[i] = imgAvgVec[i] / (matImage.cols() * matImage.rows());
            }
            double imgAvg = (imgAvgVec[0] + imgAvgVec[1] + imgAvgVec[2]) / 3;
            int brightness = -(int) ((this.contrast - 1) * imgAvg);
            matImage.convertTo(matImage, matImage.type(), this.contrast, brightness);

            Bitmap resultImage = Bitmap.createBitmap(
                this.imageData.getWidth(),
                this.imageData.getHeight(),
                this.imageData.getConfig()
            );
            Utils.matToBitmap(matImage, resultImage);
            this.imageData = resultImage;
            this.setImageBitmap(resultImage);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void reset(){
        this.contrast = 1;
        this.setImageBitmap(this.initialData);
    }
    
    public void saveImage(){
        String fileName = null;
        try{
            fileName = generateStoredFileName();
        }
        catch (Exception e){
            Log.d(TAG, "failed to create folder");
        }
        Mat matImage = new Mat();
        Utils.bitmapToMat(this.imageData, matImage);
        boolean success = Imgcodecs.imwrite(fileName, matImage);
        matImage.release();
        if(success){ 
            Log.d(TAG, "image saved, fileName: "+fileName);
            WritableMap event = Arguments.createMap();
            event.putString("fileName", fileName);
            event.putString("saveStatus", "success");
            ReactContext reactContext = (ReactContext)getContext();
            reactContext.getJSModule(
                RCTEventEmitter.class
            ).receiveEvent(
                getId(), 
                "onSave", 
                event
            );
        } else {
            WritableMap event = Arguments.createMap();
            event.putString("fileName", "");
            event.putString("saveStatus", "failure");
            ReactContext reactContext = (ReactContext)getContext();
            reactContext.getJSModule(
                RCTEventEmitter.class
            ).receiveEvent(
                getId(), 
                "onSave", 
                event
            );
        }
    }
}
