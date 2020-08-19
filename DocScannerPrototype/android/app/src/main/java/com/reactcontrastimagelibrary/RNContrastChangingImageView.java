package com.reactcontrastimagelibrary;

import android.content.Context;
import android.content.ContentResolver;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.ParcelFileDescriptor;
import android.net.Uri;

import androidx.appcompat.widget.AppCompatImageView;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.io.IOException;
import java.io.InputStream;
import java.io.FileDescriptor;
import java.io.File;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.MalformedURLException;
import java.util.concurrent.ExecutionException;

import com.facebook.react.bridge.WritableMap;

public class RNContrastChangingImageView extends AppCompatImageView {

    private Bitmap imageData = null;
    private String imageUri = null;
    private double contrast = 1;

    public RNContrastChangingImageView(Context context) {
        super(context);
    }

    public void setImageUri(String imgUri) {
        if (imgUri != this.imageUri) {
            this.imageUri = imgUri;
            Bitmap bitmap = BitmapFactory.decodeFile(imgUri);
            this.imageData = bitmap;
            this.setImageBitmap(bitmap);
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
            Utils.matToBitmap(matImage, resultImage);

            this.setImageBitmap(resultImage);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
