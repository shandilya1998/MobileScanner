import React, {Component} from 'react';
import {View,
        Image,
        requireNativeComponent} from 'react-native';
import PropTypes from 'prop-types';
import Slider from '@react-native-community/slider';

//https://productcrafters.io/blog/creating-custom-react-native-ui-components-android/

class ContrastEditor extends Component{
    constructor(props){
        super(props);
        console.log('test');
        this.state = {
            constrast : 1,
        };
        this.onValueChange = this.onValueChange.bind(this);
    }   
    
    onValueChange(value){
        this.setState({contrast : value});
    }

    onSlidingComplete(value){
        console.log(value);
    }

    render(){
        console.log('source', this.props.source.slice(7));
        return(
            <View
                style = {{
                    flex : 1,
                }}>
                <RNContrastChangingImage 
                    source = {this.props.source.slice(7)}

                    contrast = {this.state.contrast}
                    resizeMode = {'contain'}/>
                <Slider
                    minimumValue = {-1}
                    maximumValue = {3}
                    onValueChange = {
                        (value)=>{
                            this.onValueChange(value)
                        }
                    }
                    onSlidingComplete = {
                        (value)=>{
                            this.onSlidingComplete(value)
                        }
                    }/> 
            </View>
        );  
    }   
}

ContrastEditor.propTypes = {
    source: PropTypes.string.isRequired,
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
