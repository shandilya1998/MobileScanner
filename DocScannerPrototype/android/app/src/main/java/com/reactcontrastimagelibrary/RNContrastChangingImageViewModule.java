package com.reactcontrastimagelibrary;

import android.app.Activity;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;

public class RNContrastChangingImageViewModule extends ReactContextBaseJavaModule{

    public RNContrastChangingImageViewModule(ReactApplicationContext reactContext){
        super(reactContext);
    }

    @Override
    public String getName() {
        return "RNContrastChangingImageViewModule";
    }

    public Activity getActivity() {
        return this.getCurrentActivity();
    }

}
