import React, {Component} from 'react';
import {View,
        Image} from 'react-native';

//https://productcrafters.io/blog/creating-custom-react-native-ui-components-android/
// use the above link for creating the contrast editable image
// Need to create a callback for setting initial contrast value, which is to be calculated using openCV in the custom imageview component
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
