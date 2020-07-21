import React, {Component} from 'react';
import {View,
        Text, 
        Image, 
        Dimensions,
        TouchableOpacity} from 'react-native';
import {connect} from 'react-redux';
import CustomCrop from 'react-native-perspective-image-cropper';
import {styles} from '../../assets/styles';
import Icon from 'react-native-vector-icons/Ionicons';
const dimensions = Dimensions.get('window');
console.log(dimensions);

class Edit extends Component{
    constructor(props){
        super(props);
        console.log(props);
        const currentPageDimensions = {
            'width' : dimensions.width*0.9,
            'height' : dimensions.height*0.7,
            'set' : false,
        };
        this.state = {
            currentPage : props.doc[0],
            currentPageDimensions : currentPageDimensions,
            toggle : {
                crop : false,
                },
            tools : ['crop'],
        };
        this.updateImage = this.updateImage.bind(this);
        this.renderSwiperButtons = this.renderSwiperButtons.bind(this);
        this.renderHeader = this.renderHeader.bind(this);
        this.renderToolBar = this.renderToolBar.bind(this);
        this.onPressCrop = this.onPressCrop.bind(this);
        this.renderItem = this.renderItem.bind(this);
        this.crop = this.crop.bind(this);
    }

    componentDidMount(){ 
        const setDimensions = (width, height)=>{
            console.log('success');
            if(height>dimensions.height){height=dimensions.height*0.7;}
            if(width>dimensions.width){width=dimensions.width*0.9;}
            console.log(width);
            console.log(height);
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
                if(height>dimensions.height){height=dimensions.height*0.7;}
                if(width>dimensions.width){width=dimensions.width*0.9;}
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

    updateImage(image, rectCoords){
        //console.log(image);
        //console.log(rectCoords);
        const updatedPage = {
            ...this.state.currentPage,
            detectedDocument : image,
            rectCoords : rectCoords,
        };
        this.setState({currentPage : updatedPage});
    }
    
    crop(){
        this.customCrop.crop();
    }

    renderHeader(){
        return(
            <View style = {{
                    flexDirection : 'row',
                    justifyContent : 'space-between',
                    alignItems : 'center',
                    marginHorizontal : 10,
                }}>
                <View 
                    style = {styles.buttonGroup}>
                    <TouchableOpacity
                        style = {styles.button}>
                        <Icon 
                            name = 'md-more'
                            size = {40}
                            color = {'white'}
                            style={styles.buttonIcon} />
                    </TouchableOpacity>
                </View>
                <View
                    style = {styles.buttonGroup}>
                    <TouchableOpacity
                        style = {styles.button}>
                        <Icon 
                            name = 'md-done-all'
                            size = {40}
                            color = {'white'}
                            style={styles.buttonIcon} />
                        <Text style={styles.buttonText}>Done</Text>
                    </TouchableOpacity>
                </View>
            </View>
        );
    }
     
    renderSwiperButtons(){
        if(this.props.captureMultiple){
            return(
                <View
                    style = {{
                        flex : 1,
                        flexDirection : 'row',
                        justifyContent : 'space-between',
                        alignItems : 'center',
                    }}>
                    <View  
                        style={[
                            styles.buttonGroup, 
                            { marginLeft : 8 }]}>
                        <TouchableOpacity
                            style={styles.button}
                            onPress={() => {}}
                            activeOpacity={0.8}>
                            <Icon 
                                name="md-arrow-round-back" 
                                size={50} 
                                color="black" 
                                style={styles.buttonIcon}/>
                        </TouchableOpacity>
                    </View>
                    <View  
                        style={[
                            styles.buttonGroup, 
                            { marginRight: 8 }]}>
                        <TouchableOpacity
                            style={styles.button}
                            onPress={() => {}}
                            activeOpacity={0.8}>
                            <Icon 
                                name="md-arrow-round-forward" 
                                size={50} 
                                color="black" 
                                style={styles.buttonIcon} />
                        </TouchableOpacity>
                    </View>  
                </View>
            );
        }
        else{   
            return <View/>;
        }
    }

    renderToolBar(){
        //console.log(this.state.toggle.crop);
        return(
            <View 
                style = {{
                    flex : 1,
                    flexDirection : 'row',
                    justifyContent : 'center',
                    alignItems : 'center',
                }}>
                <View  
                    style={[
                        styles.buttonGroup, 
                        { 
                            backgroundColor : this.state.toggle.crop ? '#008000' :'#00000080',
                            margin: 8, 
                            zIndex : 1,
                        }]}>
                    <TouchableOpacity
                        style={styles.button}
                        onPress={()=>this.onPressCrop()} 
                        activeOpacity={0.8}>
                        <Icon 
                            name="md-crop" 
                            size={50} 
                            color={'white'} 
                            style={styles.buttonIcon} />
                        <Text style={styles.buttonText}>Crop</Text>
                    </TouchableOpacity>
                </View>   
            </View>
        );
    }

    onPressCrop(){
        console.log('pressed')
        console.log(this.state.toggle.crop);
        if(this.state.toggle.crop){
            this.crop();
        }
        else{
            const toggle = {
                'crop' : true,
            };
            console.log(toggle);
            this.setState({'toggle' : toggle});
        }
    }

    renderCropper(){
        return(
            <View 
                style = {{
                    flex : 7,
                    //marginVertical : 15,
                    flexDirection : 'column',
                    //marginHorizontal : 10,
                    //height : dimensions.height*0.7,
                    width : dimensions.width,
                    justifyContent : 'center',
                    alignSelf : 'center',
                }}>  
                <CustomCrop
                    updateImage={this.updateImage}
                    initialImage = {this.state.currentPage.originalImage}
                    height = {this.state.currentPageDimensions.height}
                    width = {this.state.currentPageDimensions.width}
                    rectangleCoordinates={this.state.currentPage.rectCoords}
                    ref={ref => (this.customCrop = ref)}
                    overlayColor="rgba(18,190,210, 1)"
                    overlayStrokeColor="rgba(20,190,210, 1)"
                    handlerColor="rgba(20,150,160, 1)"
                    enablePanStrict={true}/>
            </View>
        );
    }

    renderItem(){
        if(this.state.toggle.crop){
            return this.renderCropper();
        }
        else{
            return(
                <View style = {{
                    flex : 7,
                    //marginVertical : 15,
                    flexDirection : 'column',
                    //marginHorizontal : 10,
                    //height : dimensions.height*0.7,
                    //width : dimensions.width,
                    justifyContent : 'center',
                    alignSelf : 'center',
                }}> 
                    <Image
                        source = {{uri : this.state.currentPage.originalImage}}
                        style = {{
                            width : this.state.currentPageDimensions.width,
                            height : this.state.currentPageDimensions.height,
                        }}/>
                </View>
            );
        }
    }

    render(){ 
        //console.log(this.updateImage);
        return(
            <View 
                style = {[styles.container, {backgroundColor : 'white', paddingVertical : 10}]}>
                {this.renderHeader()}
                {this.renderItem()}
                {this.renderSwiperButtons()}
                {this.renderToolBar()}
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
