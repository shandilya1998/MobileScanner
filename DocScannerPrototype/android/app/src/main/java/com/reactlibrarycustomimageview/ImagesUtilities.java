package com.reactlibrarycustomimageview;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Canvas;

import android.util.TypedValue;
import android.view.View;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

public class ImagesUtilities {


    public  Bitmap createBitmapFromView( View view, int width, int height) {
        float cropWidth = 0;
        float cropHeight = 0;
        float factor = 1;
        float x;
        float y;
        Bitmap bitmap = Bitmap.createBitmap(view.getWidth(),
                view.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        view.draw(canvas);
        return bitmap;
    }

    private int convertDpToPixels(float dp) {
        return Math.round(TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP,
                dp, Resources.getSystem().getDisplayMetrics()));
    }
}
