/* @flow */
import React from 'react';
import {View} from 'react-native';
import Slider from '@react-native-community/slider';

import RNContrastChangingImageView from './RNContrastChangingImageView';

type Props = {
    onSave : () => void,
    onReset : () => void,
    source : string, 
    resizeMode : 'contain' | 'cover' | 'stretch',
};

class ContrastEditor extends React.Component<Props, *> {
    static defaultProps = {
        onSave : ()=>{},
        onReset : ()=>{},
        resizeMode : 'contain',
    };

    constructor(props: Props) {
        super(props);
        this.state = {
            contrast : 1,
        };
        this.onSave = this.onSave.bind(this);
        this.onReset = this.onReset.bind(this);
        this.onValueChange = this.onValueChange.bind(this);
        this.onSlidingComplete = this.onSlidingComplete.bind(this);
    }

    onSave: () => void;
    onSave(event) {
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

    onReset : () => void;
    onReset(event){
        console.log(event);
        if(event.nativeEvent.resetStatus){
            if(!this.props.onReset){
                return;
            }
            console.log('test2');
            this.props.onReset({resetStatus});
        }
    }

    onValueChange(value){
        //this.setState({contrast : value});
    }

    onSlidingComplete(value){
        this.setState({contrast : value});
    }

    render() {
        return(
            <View style = {{flex : 1}}>
                <RNContrastChangingImageView 
                    source = {this.props.source}
                    constrast = {this.state.contrast}
                    ref = {(ref)=>{this.view = ref;}}
                    style = {{
                        flex : 1,
                        ...this.props.style
                    }}
                    onSave = {this.onSave}
                    onReset = {this.onReset}/>
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

export default ContrastEditor;;
