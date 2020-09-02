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

//https://stackoverflow.com/questions/34739670/creating-custom-ui-component-for-android-on-react-native-how-to-send-data-to-js/44207488#44207488

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

type Props = {
    onSave: () => void,
    onReset: () => void,
    contrast: number, 
    source: string,
    resizeMode: 'contain' | 'cover' | 'stretch',
};

class ContrastEditor extends Component<Props, *>{
    constructor(props){
        super(props);
        this.state = {
            constrast : 1,
        };
        this.onValueChange = this.onValueChange.bind(this);
        this.onSave = this.onSave.bind(this);
        this.onReset = this.onReset.bind(this);
    }
    
    onSave(event:Event) {
        console.log(event);
        if(event.nativeEvent.fileName){
            if (!this.props.onSave) {
                return;
            }
            console.log('test');
            this.props.onSave({
                fileName: event.nativeEvent.fileName,
                saveStatus: event.nativeEvent.saveStatus,
            });
        }
    }
    
    onReset(event:Event){
        console.log(event);
        if(event.nativeEvent.resetStatus){
            if(!this.props.onReset){
                return;
            }
            console.log('test2');
            this.props.onReset({resetStatus});
        }
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
            UIManager.getViewManagerConfig('RNContrastChangingImageView').Commands.save,
            [],
        );
    }

    resetImage() {
        UIManager.dispatchViewManagerCommand(
            ReactNative.findNodeHandle(this.view),
            UIManager.getViewManagerConfig('RNContrastChangingImageView').Commands.reset,
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
                    onSave = {this.onSave}
                    onReset = {this.onReset}
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

export default ContrastEditor;
