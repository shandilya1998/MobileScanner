package com.reactcontrastimagelibrary;

import java.util.Map;
import javax.annotation.Nullable;

import android.util.Log;

import com.facebook.infer.annotation.Assertions;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.common.MapBuilder;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.uimanager.SimpleViewManager;
import com.facebook.react.uimanager.ThemedReactContext;
import com.facebook.react.uimanager.annotations.ReactProp;


public class RNContrastChangingImageManager extends SimpleViewManager<RNContrastChangingImageView> {

    public static final int COMMAND_SAVE_IMAGE = 1;
	public static final int COMMAND_RESET_IMAGE = 2;
    
    private RNContrastChangingImageModule mContextModule;    

    @Override
    public String getName() {
        return "RNContrastChangingImageView";
    }

    public RNContrastChangingImageManager(ReactApplicationContext reactContext) {
		mContextModule = new RNContrastChangingImageModule(reactContext);
	}

    @Override
    protected RNContrastChangingImageView createViewInstance(ThemedReactContext reactContext) {
        return new RNContrastChangingImageView(reactContext, mContextModule.getActivity());
    }

    @ReactProp(name = "source")
    public void setImageUri(RNContrastChangingImageView view, String imgUrl) {
        view.setImageUri(imgUrl);
    }

    @ReactProp(name = "contrast", defaultFloat = 1f)
    public void setContrastValue(RNContrastChangingImageView view, float contrast) {
        view.setContrast(contrast);
    }

    @ReactProp(name = "resizeMode")
    public void setResizeMode(RNContrastChangingImageView view, String mode) {
        view.setResizeMode(mode);
    }
    
    @Override
	public Map<String,Integer> getCommandsMap() {
		Log.d("React"," View manager getCommandsMap:");
		return MapBuilder.of(
				"saveImage",
				COMMAND_SAVE_IMAGE,
				"resetImage",
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
				view.saveImage();
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
}
