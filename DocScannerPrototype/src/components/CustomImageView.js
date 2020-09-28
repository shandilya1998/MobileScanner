import React, {Component} from 'react';
import {View, UIManager, findNodeHandle, requireNativeComponent, NativeModules} from 'react-native';
import Slider from '@react-native-community/slider';

const RNCustomImageView = requireNativeComponent(
    'RNCustomImageView',
    CustomImageView,
    {nativeOnly : {onChange : true}}
);

class CustomImageView extends Component{
    
    constructor(props){
        super(props);
        console.log(props);
        this.state = {
            'contrast' : 1,
            'saturation' : 1,
            'brightness' : 0
        };
        this.onSave = this.onSave.bind(this);
        this.onSlidingComplete = this.onSlidingComplete.bind(this);
        this.save = this.save.bind(this);
        this.onChange = this.onChange.bind(this);
    }

    onSlidingComplete(value){
        if(this.props.controlledParam == 'contrast'){
            this.setState({'contrast' : value});
        }
        else if(this.props.controlledParam == 'saturation'){
            this.setState({'saturation' : value});
        }
        else if(this.props.controlledParam == 'brightness'){
            this.setState({'brightness' : value});
        }
    }

    onChange(event){
        console.log(event);
    }

    onSave(path){
        console.log('implemented');
        console.log(path);
    }

    async save(){
        if (this.customimageview) {
            const handle = findNodeHandle(this.customimageview);      
            if (!handle) {
                throw new Error('Cannot find node handles');
            }      
            await Platform.select({
                android: async () => {
                    return UIManager.dispatchViewManagerCommand(
                        handle,
                        UIManager.RNCustomImageView.Commands.save,
                        [],
                    );
                },
                ios: async () => {
                    return NativeModules.RNCustomImageViewManager.save(handle);
                },
            })();
        } 
        else {
            throw new Error('No ref to RNCustomImageView component, check that component is mounted');
    }
    }

    render(){
        return(
            <View style = {this.props.style}>
                <RNCustomImageView
                    ref = {(ref)=>{this.customimageview = ref}}
                    brightness = {this.state.brightness}
                    contrast = {this.state.contrast}
                    saturation = {this.state.saturation}
                    src = {{ 
                        source: [{uri: this.props.source}], 
                        width: this.props.width, 
                        height: this.props.height
                    }}
                    onChange = {this.onChange}
                    style = {{flex : 1}}/>
                <Slider 
                    maximumValue={10}
                    minimumValue={-10}
                    value = {0}
                    onSlidingComplete={this.onSlidingComplete}/>
            </View>
        );
    }
}

export default CustomImageView;
