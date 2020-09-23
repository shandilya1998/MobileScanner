package com.reactlibrarycustomimageview;

import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.WritableArray;

public class Image {

    int height;
    int width;
    ReadableArray sources;

    public Image(int height, int width, ReadableArray sources) {
        this.height = height;
        this.width = width;
        this.sources = sources;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public ReadableArray getSource() {
        return sources;
    }

}
