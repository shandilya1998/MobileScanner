import React, {Component} from 'react';
import {View,
        Text, 
        Image, 
        Dimensions,
        TouchableOpacity,
        Platform} from 'react-native';
import {connect} from 'react-redux';
import CustomCrop from 'react-native-perspective-image-cropper';
import {styles} from '../../assets/styles';
import Icon from 'react-native-vector-icons/Ionicons';
const dimensions = Dimensions.get('window');
//console.log(dimensions);
let RNFS = require('react-native-fs');
const cachesDir = RNFS.CachesDirectoryPath;
const writeDir = `${cachesDir}/RNRectangleScanner/`;
//console.log(cachesDir);
import {updateDoc} from '../../actions/actions';
//console.log(Platform.OS);
class Edit extends Component{
    constructor(props){
        super(props);
        //console.log(props);
        const currentPageDimensions = {
            'width' : dimensions.width,
            'height' : dimensions.height,
            'set' : false,
        };
        this.state = {
            doc : props.doc,
            currentPage : {
                pageNum : 0,
                updated : false,
                dimensions : currentPageDimensions,
            },
            toggle : {
                crop : false,
                },
            tools : ['crop'],
            saving : false,
            detectedViewDimensions : {
                width : dimensions.width,
                height : dimensions.height,
            },
        };
        this.updateImage = this.updateImage.bind(this);
        this.renderSwiperButtons = this.renderSwiperButtons.bind(this);
        this.renderHeader = this.renderHeader.bind(this);
        this.renderToolBar = this.renderToolBar.bind(this);
        this.onPressCrop = this.onPressCrop.bind(this);
        this.renderItem = this.renderItem.bind(this);
        this.crop = this.crop.bind(this);
        this.onPressDone = this.onPressDone.bind(this);
        this.renderOverlay = this.renderOverlay.bind(this);
        this.computeDetectedViewDimensions = this.computeDetectedViewDimensions.bind(this);
        this.onPressNext = this.onPressNext.bind(this);
        this.onPressPrevious = this.onPressPrevious.bind(this);
    }

    componentDidMount(){ 
        const setDimensions = (width, height)=>{
            console.log('success');
            console.log(width);
            console.log(height);
            let {currentPage} = this.state;
            currentPage.dimensions = {
                height : height,
                width : width,
                set : true,
            };
            this.setState({
                'currentPage' : currentPage,
            });
        }; 
        Image.getSize(
            this.state.doc[this.state.currentPage.pageNum].originalImage,
            setDimensions,
            (err)=> console.log(err)
        );
        this.computeDetectedViewDimensions()
    }
    
    componentDidUpdate(){
        if(!this.state.currentPage.dimensions.set){
            const setDimensions = (width, height)=>{
                console.log('success');
                console.log(width);
                console.log(height);
                let {currentPage} = this.state;
                currentPage.dimensions = { 
                    height : height,
                    width : width,
                    set : true,
                };
                this.setState({
                    'currentPage' : currentPage,
                }); 
            };  
            Image.getSize(
                this.state.currentPage.originalImage,
                setDimensions,
                (err)=> console.log(err)
            ); 
        } 
        if(this.state.currentPage.updated){
            this.props.updateDoc(this.state.doc);
            let {currentPage} = this.state;
            currentPage.updated = false;
            this.setState({
                'currentPage' : currentPage,
            });            
            this.computeDetectedViewDimensions();
        }
    }

    async updateImage(image, rectCoords){
        console.log('code coordinates');
        console.log(rectCoords);
        const type = Platform.OS=='android'?'png':'jpeg'
        const now = Date.now()
        const writeFile = `${writeDir}page${this.state.currentPage.pageNum}_${now}.${type}`
        console.log(writeFile);
        let exists = await RNFS.exists(writeFile);
        if(exists){
            console.log('image deleted')
            await RNFS.unlink(writeFile);
        } 
        RNFS.writeFile(writeFile, image, 'base64')
        exists = await RNFS.exists(writeFile);
        if(exists){
           console.log('image saved'); 
        }
        const {doc} =  this.state;
        doc[this.state.currentPage.pageNum] = {
            ...doc[this.state.currentPage.pageNum],
            detectedDocument : `file://${writeFile}`,
            rectCoords : rectCoords,
        };
        this.setState({'doc' : doc}); 
    }
    
    crop(){
        this.customCrop.crop();
        let {currentPage} = this.state;
        currentPage.updated = true;
        this.setState({
            'currentPage' : currentPage,
        });
    }

    renderHeader(){
        return(
            <View style = {
                    styles.overlay,
                    {
                        flexDirection : 'row',
                        justifyContent : 'space-between',
                        alignItems : 'center',
                        marginHorizontal : 10,
                        marginVertical : 10,
                        paddingVertical : 10,
                    }}>
                <View 
                    style = {styles.buttonGroup}>
                    <TouchableOpacity
                        style = {styles.button}>
                        <Icon 
                            name = 'md-more'
                            size = {40}
                            color = {'white'}
                            style={[styles.buttonIcon, {fontSize:40}]} />
                    </TouchableOpacity>
                </View>
                <View
                    style = {styles.buttonGroup}>
                    <TouchableOpacity
                        onPress = {()=>this.onPressDone()}
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

    onPressDone(){
        this.props.updateDoc(this.state.doc);
        this.props.navigation.navigate('saved'); 
    }

    onPressNext(){
        console.log('next');
        if(this.state.currentPage.pageNum<this.state.doc.length-1){
            const currentPageDimensions = { 
                'width' : dimensions.width,
                'height' : dimensions.height,
                'set' : false,
            };
            const currentPage = {
                pageNum : this.state.currentPage.pageNum+1,
                updated : false,
                dimensions : currentPageDimensions,
            };
            this.setState({
                'currentPage' : currentPage,
            });
        }    
    }

    onPressPrevious(){
        if(this.state.currentPage.pageNum>0){
            const currentPageDimensions = {
                'width' : dimensions.width,
                'height' : dimensions.height,
                'set' : false,
            };
            const currentPage = {
                pageNum : this.state.currentPage.pageNum-1,
                updated : false,
                dimensions : currentPageDimensions,
            };
            this.setState({
                'currentPage' : currentPage,
            });
        }
    }
     
    renderSwiperButtons(){
        return(
            <View
                style = {
                    styles.overlay,
                    {
                        width : dimensions.width,
                        flex : 1.5,
                        flexDirection : 'row',
                        justifyContent : 'space-between',
                        alignItems : 'center',
                        paddingHorizontal : 10,
                    }}>
                <View  
                    style={[
                        styles.buttonGroup, 
                        { marginLeft : 8 }]}>
                    <TouchableOpacity
                        onPress = {()=>this.onPressPrevious()}
                        style={[
                            styles.button,
                            {
                                height : 35,
                                width : 32.5,
                            }    
                        ]}
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
                        onPress = {()=>{this.onPressNext()}}
                        style={[
                            styles.button,
                            {   
                                height : 35, 
                                width : 32.5,
                            }    
                        ]}
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

    renderToolBar(){
        //console.log(this.state.toggle.crop);
        return(
            <View 
                style = {
                    styles.overlay,
                    {
                        flex : 1.5,
                        justifyContent : 'center',
                        //position : 'absolute',
                        alignItems : 'center',
                        flexDirection : 'row',
                        paddingBottom : 10,
                        marginBottom : 10,
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
        //console.log('pressed')
        //console.log(this.state.toggle.crop);
        if(this.state.toggle.crop){
            this.crop();
            const toggle = {crop : false};
            this.setState({'toggle' : toggle});
            
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
            <View>
                <CustomCrop
                    updateImage={this.updateImage}
                    initialImage = {this.state.doc[this.state.currentPage.pageNum].originalImage}
                    height = {this.state.currentPage.dimensions.height}
                    width = {this.state.currentPage.dimensions.width}
                    rectangleCoordinates={this.state.doc[this.state.currentPage.pageNum].rectCoords}
                    ref={ref => (this.customCrop = ref)}
                    overlayColor="rgba(18,190,210, 1)"
                    overlayStrokeColor="rgba(20,190,210, 1)"
                    handlerColor="rgba(20,150,160, 1)"
                    enablePanStrict={true}/>
            </View>
        );
    }

    computeDetectedViewDimensions(){
        const setDimensions = (width, height)=>{
            console.log('success');
            console.log(width);
            console.log(height);
            const detectedViewDimensions = {
                width : dimensions.width,
                height : dimensions.width*height/width,
            };
            this.setState({
                'detectedViewDimensions' : detectedViewDimensions,
            }); 
        };  
        Image.getSize(
            this.state.doc[this.state.currentPage.pageNum].detectedDocument,
            setDimensions,
            (err)=> console.log(err)
        ); 
    }

    renderItem(){
        //console.log(this.state);
        if(this.state.toggle.crop){
            return this.renderCropper();
        }
        else{
            //this.computeDetectedViewDimensions();
            return(
                <View style = {{
                    flex : 1,
                    //marginVertical : 15,
                    flexDirection : 'column',
                    //marginHorizontal : 10,
                    //height : dimensions.height,
                    //width : dimensions.width,
                    justifyContent : 'center',
                    alignSelf : 'center',
                }}> 
                    <Image
                        source = {{
                            uri : this.state.doc[this.state.currentPage.pageNum].detectedDocument}}
                        style = {{
                            height : this.state.detectedViewDimensions.height,
                            width : this.state.detectedViewDimensions.width,
                        }}
                        resizeMode = {'contain'}/>
                </View>
            );
        }
    }
 
    renderOverlay(){
        return(
            <View
                style = {[
                    styles.overlay, 
                    {justifyContent : 'space-between'}
                ]}> 
                {this.renderHeader()}
                <View
                    style = {{
                        //flex : 1,
                        flexDirection : 'column',
                        jusitfyContent : 'space-between',
                        alignItems : 'center',
                        height : 150,
                    }}>
                    {this.props.route.params.captureMultiple?this.renderSwiperButtons():null}
                    {this.renderToolBar()}
                </View>
            </View>
        );
    }

    render(){ 
        //console.log(this.updateImage);
        return(
            <View 
                style = {[
                    styles.container, 
                    {
                        backgroundColor : 'white', 
                        paddingVertical : 0,
                    }]}>
                {this.renderItem()}
                {this.renderOverlay()}
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
        updateDoc : (doc) => dispatch(updateDoc),
    };
};

export default connect( mapStateToProps, mapDispatchToProps)(Edit);
