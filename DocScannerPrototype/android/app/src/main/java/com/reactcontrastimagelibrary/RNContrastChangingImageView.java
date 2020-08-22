package com.reactcontrastimagelibrary;

import android.content.Context;
import android.content.ContentResolver;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.ParcelFileDescriptor;
import android.net.Uri;
import android.util.Log;

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

public class RNContrastChangingImageView extends AppCompatImageView {
    public static final String TAG = "ContrastEditor";
    private String cacheFolderName = "RNContrastChangingImage";
    private Bitmap imageData = null;
    private Bitmap modifiedData = null;
    private String imageUri = null;
    private double contrast = 1;
    protected Context mContext;
    public static RNContrastChangingImageView instance = null;

    public RNContrastChangingImageView(Context context) {
        super(context);
        this.mContext = context;
    }

    public static RNContrastChangingImageView getInstance() {
        return instance;
    }

    public static void createInstance(Context context) {
        instance = new RNContrastChangingImageView(context);
    }

    public void setImageUri(String imgUri) {
        //Log.d(TAG, "set image");
        //Log.d(TAG, "image source : " + imgUri);
        //Log.d(TAG, "cache dir :" + this.mContext.getCacheDir() );
        if (imgUri != this.imageUri) {
            this.imageUri = imgUri;
            try{
                File imgFile = new File(imgUri);
                Bitmap bitmap = BitmapFactory.decodeStream(
                    new FileInputStream(imgFile)
                );
                //Log.d(TAG, "set image source");
                this.imageData = bitmap;
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
            Utils.bitmapToMat(this.imageData, matImage);

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
            this.modifiedData = resultImage;
            Utils.matToBitmap(matImage, resultImage);

            this.setImageBitmap(resultImage);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public String saveImage(){
        String fileName = null;
        try{
            fileName = generateStoredFileName();
        }
        catch (Exception e){
            Log.d(TAG, "failed to create folder");
        }
        Mat matImage = new Mat();
        Utils.bitmapToMat(this.modifiedData, matImage);
        boolean success = Imgcodecs.imwrite(fileName, matImage);
        matImage.release();
        if(success){ 
            return fileName;
        } else {
            return null;
        }
    }
}
