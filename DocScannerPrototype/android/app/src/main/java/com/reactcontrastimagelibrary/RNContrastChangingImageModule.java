package com.reactcontrastimagelibrary;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;

public class RNRectangleScannerModule extends ReactContextBaseJavaModule{

    public RNRectangleScannerModule(ReactApplicationContext reactContext){
        super(reactContext);
    }

    @Override
    public String getName() {
        return "RNContrastChangingImageManager";
    }

    @ReactMethod
    public String save(){
        RNContrastChangingImageView view = RNContrastChangingImageView.getInstance();
        String path = view.saveImage();
        return path;        
    }
}
