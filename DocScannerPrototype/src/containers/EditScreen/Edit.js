import React, {Component} from 'react';
import {View, 
        Text,
        Dimensions,
        Image,
        TouchableOpacity} from 'react-native';
import {connect} from 'react-redux';
import {styles} from '../../assets/styles';
import Carousel from 'react-native-snap-carousel';
import Icon from 'react-native-vector-icons/Ionicons';
import CustomCrop from "react-native-perspective-image-cropper";
import {updatePage} from '../../actions/actions';
import { CACHE_FOLDER_NAME } from 'react-native-rectangle-scanner';

const dimensions = Dimensions.get('window');

class Edit extends Component{
    constructor(props){
        super(props);
        console.log(CACHE_FOLDER_NAME);
        this.state = {
            'currentPage' : {}
        }
        this.updateImage = this.updateImage.bind(this);
        this.renderCarouselButtons = this.renderCarouselButtons.bind(this);
        this.getImageSpecs = this.getImageSpecs.bind(this);
        this.renderItem = this.renderItem.bind(this);
        this.onPressBack = this.onPressBack.bind(this);
        this.onPressNext = this.onPressNext.bind(this);
        this.updateImage = this.updateImage.bind(this);
        this.renderCarouselButtons = this.renderCarouselButtons.bind(this);
        this.onPressDone = this.onPressDone.bind(this);
    }

    componentDidMount(){
        try{
            this.setState({
            'currentImage' : this.props.doc[0],
            });
        }
        catch(err){
            console.log(err);
        }
    }

    onPressBack(){
        if(this.state.currentPage.pageNum>1){
            this._carousel.snapToItem(this.state.pageNum-1);
        }
        this.props.updatePage(this.state.currentPage); 
    }
    
    onPressNext(){
        if(this.state.currentPage.pageNum<this.props.doc.length-1){
            this._carousel.snapToItem(this.state.pageNum+1);
        }
    }

    updateImage(updatedImage, rectCoords){
        let currentPage = {
            ...this.state.currentPage,
            'rectCoords' : rectCoords,
            'detectedDocument' : updatedImage,                    
        };
        this.setState({
            'currentPage' : currentPage,
        });  
    }    

    renderCarouselButtons(){
        return(
            <View
                style = {{
                    flexDirection : 'row',
                    justifyContent : 'space-between',
                    width : dimensions.width*0.95,
                    height : 50, 
                    }}> 
                <View style={styles.buttonGroup}>
                    <TouchableOpacity
                        style={styles.button}
                        onPress={()=>this.onPressBack()}
                        activeOpacity={0.8}>
                        <Icon 
                            name="md-arrow-round-forward" 
                            size={40}     
                            style={styles.buttonIcon} />
                    </TouchableOpacity>
                </View>
                <View style={styles.buttonGroup}>
                    <TouchableOpacity
                        style={styles.button}
                        onPress={()=>this.onPressNext()}
                        activeOpacity={0.8}>
                        <Icon 
                            name="md-arrow-round-backward" 
                            size={40}     
                            style={styles.buttonIcon} />
                    </TouchableOpacity>
                </View>  
            </View>
        );
    }

    getImageSpecs(item){
        if(this.state.currentPage.pageNum != item.pageNum){
            this.setState({'currentPage' : item});
        }
        let image = {};
        Image.getSize(item.originalImage,
            (width, height) => {
                this.setState({'currentPage' : item.pageNum})
                image = {
                    imageWidth: width,
                    imageHeight: height,
                    initialImage: item.originalImage,
                    rectCoords: {
                    topLeft: {
                        x: item.rectCoords.topLeft.x,
                        y: item.rectCoords.topLeft.y},
                    topRight: {
                        x: item.rectCoords.topRight.x,
                        y: item.rectCoords.topRight.y },
                    bottomRight: {
                        x: item.rectCoords.bottomRight.x,
                        y: item.rectCoords.bottomRight.y,},
                    bottomLeft: {
                        x: item.rectCoords.bottomLeft.x,
                        y: item.rectCoords.bottomLeft.y, }
                    }
                },
                (err) => {console.log(err);}
            }
        );
    }

    renderItem({item, index}){
        const image = this.getImageSpecs(item);
        const nextButtons = null;
        if(this.props.captureMultiple){
            nextButtons = this.renderCarouselButtons();
        }
        return(
            <View>
                <View>
                    <CustomCrop
                        updateImage = {this.upadateImage}
                        ref={ref => (this.customCrop = ref)}
                        rectangleCoordinates = {item.rectCoords}
                        initialImage={item.initalImage}
                        height={image.imageHeight}
                        width={image.imageWidth}
                        overlayColor="rgba(18,190,210, 1)"
                        overlayStrokeColor="rgba(20,190,210, 1)"
                        handlerColor="rgba(20,150,160, 1)"
                        enablePanStrict={false}/>
                </View>
                {nextButtons}
            </View> 
        ); 
    }

    onPressDone(){
        this.props.navigation.navigate('save'); 
    }

    render(){
        return(
            <View>
                <View style={styles.buttonGroup}>
                    <TouchableOpacity
                        style={styles.button}
                        onPress={()=>this.onPressDone()}
                        activeOpacity={0.8}>
                        <Icon 
                            name="md-arrow-round-backward" 
                            size={40}     
                            style={styles.buttonIcon} />
                    </TouchableOpacity>
                </View>
                <View>
                    <Carousel
                        data = {this.props.doc}
                        renderItem = {this.renderItem}
                        sliderWidth = {dimensions.width}
                        itemWidth = {dimensions.width*0.95}
                        scrollEnabled = {false}
                        layout = {'default'}
                        ref = {ref => {this._carousel = ref;}}/>
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
    return {
        updatePage : (page) => dispatch(updatePage(page)),
    };
};

export default connect(mapStateToProps, mapDispatchToProps)(Edit);
