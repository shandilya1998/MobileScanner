package com.reactcontrastimagelibrary;

import java.util.Map;
import javax.annotation.Nullable;

import android.util.Log;
import android.content.Context;

import com.facebook.infer.annotation.Assertions;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.common.MapBuilder;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.uimanager.SimpleViewManager;
import com.facebook.react.uimanager.ThemedReactContext;
import com.facebook.react.uimanager.annotations.ReactProp;

public class RNContrastChangingImageViewManager extends SimpleViewManager<RNContrastChangingImageView> {
    private static final String REACT_CLASS = "RNContrastChangingImageView";
    private RNContrastChangingImageView imageView = null;
    private Context context;
 
    public RNContrastChangingImageViewManager(ReactApplicationContext context) {
        this.context = context;
    }

    @Override
    public String getName() {
        return REACT_CLASS;
    }

    @Override
    public RNContrastChangingImageView createViewInstance(ThemedReactContext context) {
        if (imageView == null) {
            imageView = new RNContrastChangingImageView(context);
        }
        return imageView;
    }

    @ReactProp(name = "source")
    public void setSource(RNContrastChangingImageView imageView, String source) {
        imageView.setSource(source);
    }
    
    @ReactProp(name = "contrast")
    public void setResourceType(RNContrastChangingImageView imageView, double contrast) {
        imageView.setContrast(contrast);
    }
}
