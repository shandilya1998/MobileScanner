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
        this.state = {
            constrast : 1,
        };
        this.onValueChange = this.onValueChange.bind(this);
        this.onChange = this.onChange.bind(this);
        this.subscriptions = [];
    }
    
    onChange(event) {
        if(event.nativeEvent.fileName){

            if (!this.props.onSaveEvent) {
                return;
            }
            this.props.onSaveEvent({
                pathName: event.nativeEvent.fileName,
                encoded: event.nativeEvent.saveStatus,
            });
        }
    }
 
    componentDidMount(){
        if (this.props.onSaveEvent) {
            let sub = DeviceEventEmitter.addListener(
                'onSaveEvent',
                this.props.onSaveEvent
            );
            this.subscriptions.push(sub);
        }
    }   
    
    componentWillUnmount() {
        this.subscriptions.forEach(sub => sub.remove());
        this.subscriptions = [];
    }
    
    onValueChange(value){
        this.setState({contrast : value});
    }

    onSlidingComplete(value){
        this.setState({contrast : value});
    }

    saveImage() {
        UIManager.dispatchViewManagerCommand(
            ReactNative.findNodeHandle(this),
            UIManager.getViewManagerConfig('RNContrastChangingImageView').Commands.saveImage,
            [],
        );
    }

    resetImage() {
        UIManager.dispatchViewManagerCommand(
            ReactNative.findNodeHandle(this),
            UIManager.getViewManagerConfig('RNContrastChangingImageView').Commands.resetImage,
            [],
        );
    }

    render(){
        return(
            <View
                style = {{
                    flex : 1,
                }}>
                <RNContrastChangingImageView 
                    style = {{
                        flex : 1
                    }}
                    source = {this.props.source.slice(7)}

                    contrast = {this.state.contrast}
                    resizeMode = {'contain'}
                    onChange = {this.onChange}
                    {...this.props}/>
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

let RNContrastChangingImageView = requireNativeComponent(
    'RNContrastChangingImageView', 
    ContrastEditor,
    {
       nativeOnly: { onChange: true }
    }
);

export default ContrastEditor;
