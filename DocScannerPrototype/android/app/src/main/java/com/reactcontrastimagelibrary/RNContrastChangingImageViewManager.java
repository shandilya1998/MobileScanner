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

//Tutorial on RN bridge https://itnext.io/how-to-build-react-native-bridge-and-get-pdf-viewer-44614f11e08e

public class RNContrastChangingImageViewManager extends SimpleViewManager<RNContrastChangingImageView> {

    private static final String TAG = "ContrastEditor";
    public static final int COMMAND_SAVE_IMAGE = 1;
	public static final int COMMAND_RESET_IMAGE = 2;
    private Context mContext;
    private RNContrastChangingImageView view = null;
    
    private RNContrastChangingImageViewModule mContextModule;    

    @Override
    public String getName() {
        return "RNContrastChangingImageView";
    }

    public RNContrastChangingImageViewManager(ReactApplicationContext reactContext) {
	    mContext = reactContext;
    	mContextModule = new RNContrastChangingImageViewModule(reactContext);
	}

    @Override
    protected RNContrastChangingImageView createViewInstance(ThemedReactContext reactContext) {
        
        if(view == null){
            view = new RNContrastChangingImageView(reactContext, mContextModule.getActivity());     
        }
        return view;
    }

    @ReactProp(name = "source")
    public void setImageUri(RNContrastChangingImageView view, String imgUrl) {
        view.setImageUri(imgUrl);
    }

    @ReactProp(name = "contrast", defaultFloat = 1f)
    public void setContrastValue(RNContrastChangingImageView view, float contrast) {
        Log.d(TAG, "contrast set");
        view.setContrast(contrast);
    }

    @ReactProp(name = "resizeMode")
    public void setResizeMode(RNContrastChangingImageView view, String mode) {
        Log.d(TAG, "resize mode set");
        view.setResizeMode(mode);
    }
    
    @Override
	public Map<String,Integer> getCommandsMap() {
		Log.d("React"," View manager getCommandsMap:");
		return MapBuilder.of(
				"save",
				COMMAND_SAVE_IMAGE,
				"reset",
				COMMAND_RESET_IMAGE);
	}

	@Override
	public void receiveCommand(
			RNContrastChangingImageView view,
			int commandType,
			@Nullable ReadableArray args) {
		Assertions.assertNotNull(view);
		Assertions.assertNotNull(args);
		switch (commandType) {
			case COMMAND_SAVE_IMAGE: {
				Log.d(TAG, "Command called");
                view.save();
				return;
			}
			case COMMAND_RESET_IMAGE: {
				view.reset();
				return;
			}

			default:
				throw new IllegalArgumentException(String.format(
						"Unsupported command %d received by %s.",
						commandType,
						getClass().getSimpleName()));
		}
	}

    @Override
    public @Nullable Map getExportedCustomDirectEventTypeConstants() {
        return MapBuilder.of(
                "save",
                MapBuilder.of(
                    "registrationName",
                    "onSave"
                ),
                "reset",
                MapBuilder.of(
                    "registrationName",
                    "onReset"
                )   
            );
    }
}
