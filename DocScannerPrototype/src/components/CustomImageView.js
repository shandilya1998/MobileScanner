import React, {Component} from 'react';
import {View, requireNativeComponent} from 'react-native';
import Slider from '@react-native-community/slider';

const RNCustomImageView = requireNativeComponent('RNCustomImageView');

class CustomImageView extends Component{
    constructor(props){
        super(props);
    }

    render(){
        return(
            <View>
                <RNCustomImageView/>
                <Slider/>
            </View>
        );
    }
}

export default CustomImageView;
