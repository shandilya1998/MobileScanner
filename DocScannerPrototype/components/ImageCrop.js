import React, {Component} from 'react';
import {
  Platform,
  StyleSheet,
  Text,
  View,
  Image,
  TouchableOpacity,
} from 'react-native';
import CustomCrop from 'react-native-perspective-image-cropper';

export default class CropView extends Component {
  componentWillMount() {
    const {image} = this.props;
    console.log(image);
    Image.getSize(image, (width, height) => {
      this.setState({
        imageWidth: width,
        imageHeight: height,
        initialImage: image,
        rectangleCoordinates: {
          topLeft: {x: 10, y: 10},
          topRight: {x: 10, y: 10},
          bottomRight: {x: 10, y: 10},
          bottomLeft: {x: 10, y: 10},
        },
      });
    });
  }

  updateImage(newCoordinates) {
    const {image} = this.props;
    this.setState({
      image: image,
      rectangleCoordinates: newCoordinates,
    });
  }

  crop() {
    this.customCrop.crop();
  }

  render() {
    const {
      rectangleCoordinates,
      initialImage,
      imageWidth,
      imageHeight,
    } = this.state;
    const {image} = this.props;
    return (
      <View>
        {image && rectangleCoordinates ? (
          <CustomCrop
            updateImage={this.updateImage}
            rectangleCoordinates={rectangleCoordinates}
            initialImage={initialImage}
            height={imageHeight}
            width={imageWidth}
            ref={(ref) => (this.customCrop = ref)}
            overlayColor="rgba(18,190,210, 1)"
            overlayStrokeColor="rgba(20,190,210, 1)"
            handlerColor="rgba(20,150,160, 1)"
          />
        ) : null}
        <TouchableOpacity onPress={this.crop}>
          <Text>CROP IMAGE</Text>
        </TouchableOpacity>
      </View>
    );
  }
}
