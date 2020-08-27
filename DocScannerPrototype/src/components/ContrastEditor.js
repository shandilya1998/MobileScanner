import React, {Component} from 'react';
let ReactNative = require('react-native');
let {View,
     Image,
     requireNativeComponent,
     DeviceEventEmitter,
     UIManager,
     ViewPropTypes} = ReactNative;
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
        this._onSave = this._onSave.bind(this);
        this._onReset = this._onReset.bind(this);
    }
    
    _onSave(event) {
        console.log(event);
        if(event.nativeEvent.fileName){

            if (!this.props.onSaveSuccess) {
                return;
            }
            console.log('test');
            this.props.onSaveSuccess({
                fileName: event.nativeEvent.fileName,
                saveStatus: event.nativeEvent.saveStatus,
            });
        }
    }
    
    _onReset(event){
        console.log(event);
    }
 
    componentDidMount(){
        //console.log(this.props.source);
    }   
    
    componentWillUnmount() {
    }
    
    onValueChange(value){
        //this.setState({contrast : value});
    }

    onSlidingComplete(value){
        this.setState({contrast : value});
    }

    saveImage() {
        UIManager.dispatchViewManagerCommand(
            ReactNative.findNodeHandle(this.view),
            UIManager.getViewManagerConfig('RNContrastChangingImageView').Commands.saveImage,
            [],
        );
    }

    resetImage() {
        UIManager.dispatchViewManagerCommand(
            ReactNative.findNodeHandle(this.view),
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
                    ref = {(ref)=>{this.view = ref;}}
                    style = {{
                        flex : 1
                    }}
                    source = {this.props.source}

                    contrast = {this.state.contrast}
                    resizeMode = {'contain'}
                    onSave = {this._onSave}
                    onReset = {this._onReset}
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
    onSave: PropTypes.func,
    onReset : PropTypes.func,
    source: PropTypes.string.isRequired,
    resizeMode: PropTypes.oneOf(['contain', 'cover', 'stretch']),
}

ContrastEditor.defaultProps = {
    resizeMode: 'contain',
    onSave : ()=>{},
    onReset : ()=>{},
}

const componentInterface = {
 name: 'RNContrastChangingImageView',
 propTypes: {
    ...ViewPropTypes,
    onSave : PropTypes.func,
    onReset : PropTypes.func,
    source : PropTypes.string,
    contrast : PropTypes.number,
    resizeMode: PropTypes.oneOf(['contain', 'cover', 'stretch']),    
 },
};

let RNContrastChangingImageView = requireNativeComponent(
    'RNContrastChangingImageView', 
    componentInterface,
);

export default ContrastEditor;
