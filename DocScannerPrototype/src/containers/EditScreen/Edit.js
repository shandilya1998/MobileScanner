import React, {Component} from 'react';
import {View, Image, Dimensions} from 'react-native';
import {connect} from 'react-redux';
import CustomCrop from 'react-native-perspective-image-cropper';

const dimensions = Dimensions.get('window');

class Edit extends Component{
    constructor(props){
        super(props);
        console.log(props);
        const currentPageDimensions = {
            'width' : dimensions.width*0.8,
            'height' : dimensions.height*0.8,
            'set' : false,
        };
        this.state = {
            currentPage : props.doc[0],
            currentPageDimensions : currentPageDimensions,
        };
    }

    componentDidMount(){ 
        const setDimensions = (width, height)=>{
            console.log('success');
            console.log(width);
            console.log(height);
            if(height>dimensions.height){height=dimensions.height*0.8;}
            if(width>dimensions.width){width=dimensions.width*0.8;}
            this.setState({
                currentPageDimensions : {
                    height : height,
                    width : width,
                    set : true,
                },
            }); 
        }; 
        Image.getSize(
            this.state.currentPage.originalImage,
            setDimensions,
            (err)=> console.log(err)
        );
    }
    
    componentDidUpdate(){
        if(!this.state.currentPageDimensions.set){
            const setDimensions = (width, height)=>{
                console.log('success');
                console.log(width);
                console.log(height);
                if(height>dimensions.height){height=dimensions.height*0.8;}
                if(width>dimensions.width){width=dimensions.width*0.8;}
                this.setState({
                    currentPageDimensions : {
                        height : height,
                        width : width,
                        set : true,
                    },  
                }); 
            };  
            Image.getSize(
                this.state.currentPage.originalImage,
                setDimensions,
                (err)=> console.log(err)
            ); 
        } 
    }

    render(){ 
        return(
            <View>
                <View style = {{
                    height : dimensions.height*0.85,
                    width : dimensions.width*0.85,
                }}>
                    <CustomCrop
                        initialImage = {this.state.currentPage.originalImage}
                        height = {this.state.currentPageDimensions.height}
                        width = {this.state.currentPageDimensions.width}
                        rectangleCoordinates={this.state.currentPage.rectCoords}
                        ref={ref => (this.customCrop = ref)}
                        overlayColor="rgba(18,190,210, 1)"
                        overlayStrokeColor="rgba(20,190,210, 1)"
                        handlerColor="rgba(20,150,160, 1)"
                        enablePanStrict={false}/>
                </View>
            </View>
        );
    }
}

const mapStateToProps = (state) => {
    const {scannedDocument} = state;
    return {
        doc : scannedDocument,
    };
};

const mapDispatchToProps = (dispatch) => {
    return {};
};

export default connect( mapStateToProps, mapDispatchToProps)(Edit);
