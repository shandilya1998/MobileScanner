package com.reactcontrastimagelibrary;

import android.app.Activity;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;

public class RNContrastChangingImageModule extends ReactContextBaseJavaModule{

    public RNContrastChangingImageModule(ReactApplicationContext reactContext){
        super(reactContext);
    }

    @Override
    public String getName() {
        return "RNContrastChangingImageModule";
    }

    public Activity getActivity() {
        return this.getCurrentActivity();
    }

}
