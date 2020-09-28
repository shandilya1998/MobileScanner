package com.reactlibrarycustomimageview;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.os.Environment;
import android.util.Log;

import androidx.annotation.Nullable;

import com.facebook.drawee.controller.AbstractDraweeControllerBuilder;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactContext;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.uimanager.events.RCTEventEmitter;
import com.facebook.react.views.image.GlobalImageLoadListener;
import com.facebook.react.views.image.ReactImageView;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;

public class RNCustomImageView extends ReactImageView {

    private Image sourceImage;
    private ImagesUtilities imgUtilities;
    private StorageSave saver;
    private String TAG="RNCustomImageView";

    public RNCustomImageView(Context context, AbstractDraweeControllerBuilder draweeControllerBuilder, @Nullable GlobalImageLoadListener globalImageLoadListener, @Nullable Object callerContext) {
        super(context, draweeControllerBuilder, globalImageLoadListener, callerContext);
        imgUtilities = new ImagesUtilities();
        saver = new StorageSave();
    }

    public void setImageSource(Image img) {
        this.sourceImage = img;
        this.setSource(img.getSource());
    }


    public void triggerOnSaveEvent(String uri) {
        Log.d(TAG, "called");
        Log.d(TAG, "path: "+uri);
        WritableMap args = Arguments.createMap();
        args.putString("uri",uri);
        ((ReactContext)getContext()).getJSModule(RCTEventEmitter.class).receiveEvent(getId(),"topChange",args);
    }


    private ColorMatrixColorFilter createMatrix(float brightness, float contrast, float saturation) {
        ColorMatrix satMatrix = new ColorMatrix();
        satMatrix.setSaturation(saturation);
        float[] satValues = satMatrix.getArray();

        ColorMatrix cm = new ColorMatrix();
        cm.set(new float[] {
                satValues[0]*contrast, satValues[1]*contrast, satValues[2]*contrast, satValues[3]*contrast,satValues[4]*contrast + brightness,
                satValues[5]*contrast, satValues[6]*contrast, satValues[7]*contrast, satValues[8]*contrast,satValues[9]*contrast + brightness,
                satValues[10]*contrast, satValues[11]*contrast, satValues[12]*contrast, satValues[13]*contrast,satValues[14]*contrast + brightness,
                satValues[15], satValues[16], satValues[17], satValues[18], satValues[19] });

        return new ColorMatrixColorFilter(cm);
    }

    public void applyFilter(float brightness, float contrast, float saturation) {
        ColorMatrixColorFilter matrix = createMatrix(brightness, contrast,  saturation);
        this.setColorFilter(matrix);
    }

    public void saveImageToStorage() {
        this.getDrawable();
        Bitmap bmp = imgUtilities.createBitmapFromView(this, this.sourceImage.getWidth(),this.sourceImage.getHeight());
        String path = saver.saveToExternalStorage(bmp);
        this.triggerOnSaveEvent(path);
    }
}
