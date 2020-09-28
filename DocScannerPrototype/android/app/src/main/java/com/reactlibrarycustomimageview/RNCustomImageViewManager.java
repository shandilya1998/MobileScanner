package com.reactlibrarycustomimageview;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.facebook.drawee.backends.pipeline.Fresco;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.common.MapBuilder;
import com.facebook.react.uimanager.SimpleViewManager;
import com.facebook.react.uimanager.ThemedReactContext;
import com.facebook.react.uimanager.ViewProps;
import com.facebook.react.uimanager.annotations.ReactProp;
import com.facebook.react.views.image.ImageResizeMode;

import java.util.ArrayList;
import java.util.Map;

public class RNCustomImageViewManager extends SimpleViewManager<RNCustomImageView> {

    private static final int COMMAND_SAVE = 1;
    ReactApplicationContext mCallerContext;
    float currentBright;
    float currentSat;
    float currentContrast;
    ImagesUtilities imgSaver;
    RNCustomImageView view;


    public RNCustomImageViewManager(ReactApplicationContext context) {
        mCallerContext = context;
        currentBright = 0;
        currentContrast = 1;
        currentSat = 1;
    }

    @NonNull
    @Override
    public String getName() {
        return "RNCustomImageView";
    }

    @NonNull
    @Override
    protected RNCustomImageView createViewInstance(@NonNull ThemedReactContext reactContext) {
        return new RNCustomImageView(reactContext, Fresco.newDraweeControllerBuilder(), null, mCallerContext);
    }

    /*
    @Nullable@Override
    public Map getExportedCustomDirectEventTypeConstants(){
        return MapBuilder.builder()
                .put(
                        "onSave",
                        MapBuilder.of(
                                "registrationName",
                                "onSave"
                        )
                    )
                .build();
    }*/

    @ReactProp(name = "src")
    public void setSrc(RNCustomImageView view, @Nullable ReadableMap image) {
        Image img = new Image(image.getInt("height"),image.getInt("width"),image.getArray("source"));
        view.setImageSource(img);
    }

    @ReactProp(name = "brightness")
    public void setBrightness(RNCustomImageView view, @Nullable float brightness) {
        currentBright = brightness;
        view.applyFilter(brightness,currentContrast,currentSat);
    }

    @ReactProp(name = "saturation")
    public void setSaturation(RNCustomImageView view, @Nullable float saturation) {
        currentSat = saturation;
        view.applyFilter(currentBright,currentContrast,saturation);
    }

    @ReactProp(name = "contrast")
    public void setContrast(RNCustomImageView view, @Nullable float contrast) {
        currentContrast = contrast;
        view.applyFilter(currentBright, contrast, currentSat);
    }

    @ReactProp(name = ViewProps.RESIZE_MODE)
    public void setResizeMode(RNCustomImageView view, @Nullable String resizeMode) {
        view.setScaleType(ImageResizeMode.toScaleType(resizeMode));
    }
    
    /*
    @ReactProp(name = "saveImage")
    public void saveImage(RNCustomImageView view, @Nullable boolean save) {
        if (save) {
            view.saveImageToStorage();
        }
    }
    */

    @Override
    public Map<String,Integer> getCommandsMap() {
        return MapBuilder.of("save", COMMAND_SAVE);
    }

    public void receiveCommand(final RNCustomImageView view, int command, final ReadableArray args) {
        switch (command) {
            case COMMAND_SAVE: {
                view.saveImageToStorage();
                break;
            }
            default: {
                break;
            }
        }
    }
}
