import React, {Component} from 'react';
import {View,
        Image,
        requireNativeComponent} from 'react-native';
import PropTypes from 'prop-types';

//https://productcrafters.io/blog/creating-custom-react-native-ui-components-android/

class ContrastEditor extends Component{
    constructor(props){
        super(props);
    }   

    render(){
        return(
            <View>
            </View>
        );  
    }   
}

ContrastEditor.propTypes = {
    source: PropTypes.string.isRequired,
    contrast: PropTypes.number.isRequired,
    resizeMode: PropTypes.oneOf(['contain', 'cover', 'stretch']),
}

ContrastEditor.defaultProps = {
    resizeMode: 'contain',
}

let RNContrastChangingImage = requireNativeComponent(
    'RNContrastChangingImage', 
    ContrastEditor
);

export default ContrastEditor;
