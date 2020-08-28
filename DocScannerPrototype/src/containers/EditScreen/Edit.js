import React, {Component} from 'react';
import {View,
        Text,
        SafeAreaView, 
        Image, 
        ActivityIndicator,
        Dimensions,
        TouchableOpacity,
        Platform} from 'react-native';
import {connect} from 'react-redux';
//import CustomCrop from 'react-native-perspective-image-cropper';
import {styles} from '../../assets/styles';
import Icon from 'react-native-vector-icons/Ionicons';
const dimensions = Dimensions.get('window');
//console.log(dimensions);
let RNFS = require('react-native-fs');
const cachesDir = RNFS.CachesDirectoryPath;
const writeDir = `${cachesDir}/RNRectangleScanner/`;
//console.log(cachesDir);
import {updateDoc,
        flushDoc,} from '../../actions/actions';
//console.log(Platform.OS);
import Cropper from '../../components/Cropper';
import ToolBar from '../../components/ToolBar';
import ContrastEditor from '../../components/ContrastEditor';

class Edit extends Component{
    constructor(props){
        super(props);
        //console.log(props);
        const currentPageDimensions = {
            'width' : dimensions.width,
            'height' : dimensions.height,
            'set' : false,
        };
        
        const tools =  [ 
            {   
                key : '0',
                name : 'Brush',
                icon : 'md-brush',
                onPress : ()=>{},
            },  
            {   
                key : '1',
                name : 'Brightness',
                icon : 'md-sunny',
                onPress : ()=>{},
            },  
            {   
                key : '2',
                name : 'Crop',
                icon : 'md-crop',
                onPress : ()=>this.onPressTool('crop')
            },  
            {   
                key : '3',
                name : 'Contrast',
                icon : 'md-contrast',
                onPress : ()=>this.onPressTool('contrast'),
            },  
            {   
                key : '4',
                name : 'Filter',
                icon : 'ios-color-filter',
                onPress : ()=>{}
            }   
        ]; 
        
        this.state = {
            doc : props.doc,
            currentPage : {
                pageNum : 0,
                updated : false,
                dimensions : currentPageDimensions,
            },
            loading : true,
            cropper : {
                set : false,
                minX : 0,
                minY : 0,
                maxX : dimensions.width,
                maxY : dimensions.height,
            },
            contrast : {
                set : false,
                minX : 0,
                minY : 0,
                maxX : dimensions.width,
                maxY : dimensions.height,
            },
            toggle : {
                crop : false,
                contrast : false,
                },
            tools : ['crop', 'contrast'],
            saving : false,
            detectedViewDimensions : {
                width : dimensions.width,
                height : dimensions.height,
            },
            tools : tools,
            currentTool : {
                item : tools[(tools.length-1)/2], 
                index : (tools.length-1)/2
            },
            preview : true,
        };
        this.modifyContrast = this.modifyContrast.bind(this);
        this.updateImage = this.updateImage.bind(this);
        this.renderSwiperButtons = this.renderSwiperButtons.bind(this);
        this.renderHeader = this.renderHeader.bind(this);
        this.renderToolBar = this.renderToolBar.bind(this);
        this.renderTool = this.renderTool.bind(this);
        this.onPressTool = this.onPressTool.bind(this);
        this.renderItem = this.renderItem.bind(this);
        this.crop = this.crop.bind(this);
        this.onPressDone = this.onPressDone.bind(this);
        this.renderFooter = this.renderFooter.bind(this);
        this.computeDetectedViewDimensions = this.computeDetectedViewDimensions.bind(this);
        this.onPressNext = this.onPressNext.bind(this);
        this.onPressPrevious = this.onPressPrevious.bind(this);
        this.onSnapToTool = this.onSnapToTool.bind(this);
        this.renderContrastEditor = this.renderContrastEditor.bind(this);
        this.onSaveContrastChangedImage = this.onSaveContrastChangedImage.bind(this);
    }

    componentDidMount(){ 
        const setDimensions = (width, height)=>{
            //console.log('success');
            //console.log(width);
            //console.log(height);
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
                //console.log('success');
                //console.log(width);
                //console.log(height);
                this.setState(({currentPage}) => ({
                    'currentPage' : {
                        ...currentPage,
                        'dimensions' : {
                            'set' : true,
                            'height' : height,
                            'width' : width,  
                        }
                    },
                })); 
            };  
            Image.getSize(
                this.state.currentPage.originalImage,
                setDimensions,
                (err)=> console.log(err)
            ); 
        } 
        if(this.state.currentPage.updated){
            this.props.updateDoc(this.state.doc);
            this.setState(({currentPage})=>(
                {
                    'currentPage' : {
                        ...currentPage,
                        'updated' : false,
                    },
                    'loading' : true,
                }));            
            this.computeDetectedViewDimensions();
        }
    }

    async updateImage(image, rectCoords){
        //console.log('rectangle coordinates');
        //console.log(rectCoords);
        const type = Platform.OS=='android'?'png':'jpeg'
        const now = Date.now()
        const writeFile = `${writeDir}page${this.state.currentPage.pageNum}_${now}.${type}`
        //console.log(writeFile);
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
        this.setState(({currentPage})=>({
            'currentPage' : {
                ...currentPage,
                updated : true
            },
        }));
    }

    onPressDelete(){
        this.props.navigation.navigate('scan');
        this.props.flush();

    }

    renderHeader(){
        return(
            <View style = {
                    {
                        flex : 0.5,
                        flexDirection : 'row',
                        justifyContent : 'space-between',
                        alignItems : 'center',
                        margin : 5,
                        padding : 5,
                        backgroundColor : 'blue'
                    }}
                >
                <View style = {{
                    }}>
                    <View 
                        style = {
                            styles.buttonGroup
                        }>
                        <TouchableOpacity
                            style = {[
                                styles.button,
                                {   
                                    height : 35, 
                                    width : 32.5
                                }   
                            ]}>
                            <Icon 
                                name = 'md-more'
                                size = {40}
                                color = {'white'}
                                style={[
                                    styles.buttonIcon, 
                                    {fontSize:40}]}/>
                        </TouchableOpacity>
                    </View>
                </View>
                <View
                    style = {{
                        flexDirection : 'row',
                    }}>
                    <View
                        style = {[
                            styles.buttonGroup,
                            {
                                marginHorizontal : 5,
                                //paddingHorizontal : 5,
                            }
                        ]}>
                        <TouchableOpacity
                            onPress = {()=>this.onPressDelete()}
                            style = {[
                                styles.button,
                                { 
                                    height : 35,
                                    width : 32.5
                                } 
                            ]}>
                            <Icon
                                name = 'md-trash'
                                size = {40}
                                color = {'white'}
                                style={styles.buttonIcon} />
                        </TouchableOpacity>
                    </View>
                    <View
                        style = {[
                            styles.buttonGroup,
                            {
                                marginLeft : 5,
                            }
                        ]}>
                        <TouchableOpacity
                            onPress = {()=>this.onPressDone()}
                            style = {[
                                styles.button,
                                {   
                                    height : 35, 
                                    width : 32.5
                                }   
                            ]}>
                            <Icon 
                                name = 'md-done-all'
                                size = {40}
                                color = {'white'}
                                style={styles.buttonIcon} />
                        </TouchableOpacity>
                    </View>
                </View>
            </View>
        );
    }

    onPressDone(){
        this.props.updateDoc(this.state.doc);
        this.props.navigation.navigate('saved'); 
    }

    onPressNext(){
        //console.log('next');
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
                'cropper' : {
                    'minX' : 0,
                    'minY' : 0,
                    'maxX' : dimensions.width,
                    'maxY' : dimensions.height,
                    'set' : false,
                },
                'contrast' : {
                    'set' : false,
                }
            });
        }    
    }

    onPressPrevious(){
        //console.log('previous');
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
                'cropper' : { 
                    'minX' : 0,
                    'minY' : 0,
                    'maxX' : dimensions.width,
                    'maxY' : dimensions.height,
                    'set' : false,
                },
                'contrast' : {
                    'set' : false,
                }
            });
        }
    }
     
    renderSwiperButtons(){
        return(
            <View
                style = {
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
                        onPress = {()=>{this.onPressPrevious()}}
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

    renderTool({item, index}){
        return(
            <View
                style = {{alignItems : 'center'}}>
                <View  
                    style={[
                    styles.buttonGroup, 
                    {   
                        backgroundColor : this.state.preview?'#00000080':'#008000',
                        margin : 10,  
                        zIndex : 1,
                    }]}>
                    <TouchableOpacity
                        style={styles.button}
                        onPress={item.onPress} 
                        activeOpacity={0.8}>
                        <Icon 
                            name={item.icon} 
                            size={50} 
                            color={'white'} 
          e                 style={styles.buttonIcon} />
                        <Text style={styles.buttonText}>{item.name}</Text>
                    </TouchableOpacity>
                </View>  
            </View>
        );
    }

    onSnapToTool(index){
        this.setState({
            currentTool : {
                item : this.state.tools[index],
                index : index,
            }        
        });
    }

    renderToolBar(){
        //console.log(this.state.toggle.crop);
        return(
            <View 
                style = {
                    {
                        flex : 1.5,
                        justifyContent : 'center',
                        //position : 'absolute',
                        alignItems : 'center',
                        flexDirection : 'row',
                        paddingBottom : 10,
                        marginBottom : 10,
                    }
                }>
                <ToolBar
                    renderItem = {this.renderTool}
                    tools = {this.state.tools}
                    width = {dimensions.width}
                    margin = {10}
                    containerStyle = {{
                        //backgroundColor : 'red', 
                        justifyContent : 'center',    
                        alignItems : 'center',
                    }}
                    onSnapToItem = {this.onSnapToTool}/>
            </View>
        );
    }

    modifyContrast(){
        console.log('pressed');
        this.contrastEditor.saveImage();
    }

    onPressTool(tool){
        if(tool=='crop'){
            if(this.state.toggle.crop){
                this.crop();
                const toggle = {crop : false, contrast : false};
                this.setState({toggle : toggle, preview : true});
            }   
            else{
                const toggle = { 
                    crop : true,
                    contrast : false,
                };  
                this.setState({'toggle' : toggle, preview : false});
            } 
        }
        else if(tool == 'contrast'){
            if(this.state.toggle.contrast){
                this.modifyContrast();
                const toggle = {crop : false, contrast : false};
                this.setState({toggle : toggle, preview : true});
            }   
            else{
                const toggle = { 
                    crop : false,
                    contrast : true,
                };  
                this.setState({'toggle' : toggle, preview : false});
            } 
        }
    }
    
    onSaveContrastChangedImage({fileName, saveStatus}){
        console.log(saveStatus);
        console.log(fileName);
    }

    onResetContrastChangedImage({resetStatus}){
        console.log("reset status", resetStatus);
    }

    renderContrastEditor(){
        //console.log('test2');
        return(
            <View
                style = {{
                    //flex : 8,
                    //marginVertical : 15,
                    flexDirection : 'column',
                    //marginHorizontal : 10,
                    height : dimensions.height*0.8,
                    width : dimensions.height*0.8*this.state.currentPage.dimensions.width/this.state.currentPage.dimensions.height,
                    margin : 5,
                    //padding : 5,
                    backgroundColor : 'yellow',
                    justifyContent : 'center',
                    alignSelf : 'center',
                }}
                onLayout = {(event)=>{
                    if(!this.state.contrast.set){
                        this.setState({
                            'contrast' : {
                                'minX' : event.nativeEvent.layout.x,
                                'minY' : event.nativeEvent.layout.y,
                                'maxX' : event.nativeEvent.layout.x+event.nativeEvent.layout.width,
                                'maxY' : event.nativeEvent.layout.y + event.nativeEvent.layout.height,
                                'set' : true,
                            }
                        })
                    }
                }}>
                <ContrastEditor
                    onSave = {this.onSaveContrastChangedImage}
                    onReset = {this.onResetContrastChangedImage}
                    ref = {(ref)=>{this.contrastEditor = ref;}}
                    source = {this.state.doc[this.state.currentPage.pageNum].detectedDocument.slice(7)}/>
            </View>
        );
    }

    renderCropper(){
        return(
            <View
                style = {{
                    //flex : 8,
                    //marginVertical : 15,
                    flexDirection : 'column',
                    //marginHorizontal : 10,
                    height : dimensions.height*0.8,
                    width : dimensions.height*0.8*this.state.currentPage.dimensions.width/this.state.currentPage.dimensions.height,
                    margin : 5, 
                    //padding : 5,
                    backgroundColor : 'red',
                    justifyContent : 'center',
                    alignSelf : 'center',
                }}
                onLayout = {(event)=>{
                    if(!this.state.cropper.set){
                        this.setState({
                            'cropper' : {
                                'minX' : event.nativeEvent.layout.x,
                                'minY' : event.nativeEvent.layout.y,
                                'maxX' : event.nativeEvent.layout.x+event.nativeEvent.layout.width,
                                'maxY' : event.nativeEvent.layout.y + event.nativeEvent.layout.height,
                                'set' : true,
                            }
                        })  
                    } 
                }}>
                <Cropper 
                    updateImage = {this.updateImage}
                    ref = {ref => {this.customCrop = ref}}
                    initialImage = {this.state.doc[this.state.currentPage.pageNum].originalImage}
                    viewHeight = {dimensions.height*0.8}
                    viewWidth = {dimensions.height*0.8*this.state.currentPage.dimensions.width/this.state.currentPage.dimensions.height} 
                    viewPadding = {5}
                    rectangleCoordinates = {this.state.doc[this.state.currentPage.pageNum].rectCoords}
                    minX = {this.state.cropper.minX}
                    minY = {this.state.cropper.minY}
                    maxX = {this.state.cropper.maxX}
                    maxY = {this.state.cropper.maxY}
                    height = {this.state.currentPage.dimensions.height}
                    width = {this.state.currentPage.dimensions.width}/>
            </View>
        );
    }

    computeDetectedViewDimensions(){
        const setDimensions = (width, height)=>{
            //console.log('success');
            //console.log(width);
            //console.log(height);
            const detectedViewDimensions = {
                width : dimensions.height*0.8*width/height,
                height : dimensions.height*0.8,
            };
            this.setState({
                'detectedViewDimensions' : detectedViewDimensions,
                'loading' : false,
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
        else if(this.state.toggle.contrast){
            return this.renderContrastEditor();
        }
        else{
            //this.computeDetectedViewDimensions();
            return(
                <View style = {{
                    //flex : 8,
                    //marginVertical : 15,
                    flexDirection : 'column',
                    //marginHorizontal : 10,
                    height : this.state.detectedViewDimensions.height,
                    width : this.state.detectedViewDimensions.width,
                    justifyContent : 'center',
                    alignSelf : 'center',
                    margin : 8,
                    //padding : 5,
                    backgroundColor : 'red', 
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

    renderScreenOverlay(){
        //console.log('overlay');
        let loadingState = null;
        if (this.state.loading) {
            //console.log('this');
            loadingState = (
                <View
                    style={[
                        styles.overlay,
                        {
                            backgroundColor : 'black',
                            //width : Dimensions.get('window').width, 
                        }]}>
                    <View style={styles.loadingContainer}>
                        <ActivityIndicator color="white" />
                        <Text style={styles.loadingCameraMessage}>Loading Image</Text>
                    </View>
                </View>
            );
        }
        return (
            <SafeAreaView
                style = {[
                    styles.overlay,
                ]}>
                {loadingState}
            </SafeAreaView>
        );
    }    
 
    renderFooter(){
        if(this.state.toggle.crop || this.state.toggle.contrast){
            return(
                <View
                    style = {[
                        {
                            flex : 1.5,
                            jusitifyContent : 'flex-end',
                        }
                    ]}>
                    {this.renderTool(this.state.currentTool)}
                </View>
            );
        }
        else{
            return(
                <View
                    style = {[ 
                        {
                            flex : 1.5,
                            justifyContent : 'flex-end',
                            //backgroundColor : 'green',
                        }
                    ]}>
                    {this.props.route.params.captureMultiple?this.renderSwiperButtons():null}
                    {this.renderToolBar()}
                </View>
            );
        }
    }

    render(){ 
        //console.log(this.state);
        return(
            <View 
                style = {[
                    styles.container, 
                    {
                        backgroundColor : 'white', 
                        paddingVertical : 0,
                    }]}>
                {this.renderHeader()}
                {this.renderItem()}
                {this.renderFooter()}
                {this.renderScreenOverlay()}
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
        updateDoc : (doc) => {dispatch(updateDoc(doc))},
        flush : () => {dispatch(flushDoc())}
    };
};

export default connect( mapStateToProps, mapDispatchToProps)(Edit); 
