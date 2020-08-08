import React, {Component} from 'react';
import {View, 
        Text, 
        Dimensions, 
        TextInput, 
        Image,
        TouchableOpacity} from 'react-native';
import {connect} from 'react-redux';
import RNImageToPdf from 'react-native-image-to-pdf';
let RNFS = require('react-native-fs');
const DocDir = RNFS.DownloadDirectoryPath+'/MobScanner';
//console.log(DocDir);
const dimensions = Dimensions.get('window');
import {styles} from '../../assets/styles';
import Icon from 'react-native-vector-icons/Ionicons';
import {flushDoc} from '../../actions/actions';
import Library from '../../components/Library';
import Reader from '../../components/Reader';

class Saved extends Component{
    constructor(props){
        super(props);
        //console.log(props);
        this.state = {
            preview : this.props.doc[0].detectedDocument,
            fileName : 'MobScannerPDF.pdf',
            detectedViewDimensions : {
                width : dimensions.width*0.35,
                height : dimensions.height*0.35,
                set : false,
            },
            PDF : true,
            Image : false,
            mode : 'saver',
            readerSource : {
                uri : undefined,
                cache : true
            }
        };
        this.onPressDone = this.onPressDone.bind(this);
        this.computeDetectedViewDimensions = this.computeDetectedViewDimensions.bind(this);
        this.onPressPDF = this.onPressPDF.bind(this);
        this.onPressImage = this.onPressImage.bind(this);
        this.renderPreview = this.renderPreview.bind(this);
        this.renderLibrary = this.renderLibrary.bind(this);
        this.onPressPDFFile = this.onPressPDFFile.bind(this);
    }

    componentDidMount(){
        this.computeDetectedViewDimensions();
    }

    componentDidUpdate(){
        if(!this.state.detectedViewDimensions.set){
            this.computeDetectedViewDimensions();
        }
    }
    
    async convertToPDF(){
        let imagePaths = [];
        let i;
        for(i = 0; i<this.props.doc.length; i++){
            imagePaths.push(this.props.doc[i].detectedDocument.slice(7));
        }
        try {
		    const options = {
			    imagePaths: imagePaths,
			    name: this.state.fileName,
			    maxSize: { // optional maximum image dimension - larger images will be resized
				    width: 900,
				    height: Math.round(900*dimensions.height / dimensions.width),
			    },
			    quality: .7, // optional compression paramter
		    };
		    const pdf = await RNImageToPdf.createPDFbyImages(options);
		
		    //console.log(pdf.filePath);
            const DestPath = `${DocDir}/${this.state.fileName}`;
            //console.log(DestPath);
            const exists = await RNFS.exists(DestPath);
            if(exists){
                //console.log('delete');
                await RNFS.unlink(DestPath);
            }
            await RNFS.copyFile(pdf.filePath, DestPath ).then(
            ()=>console.log('done successfully!'),
            ()=>console.log('error'));
        } catch(e) {
		    console.log(e);
	    }
    }

    onPressDone(){
        this.convertToPDF();
    }
    
    onPressClose(){
        this.props.navigation.navigate('scan'); 
        this.props.flush();
    }

    onPressPDFFile(item){
        this.setState({
            'mode' : 'reader',
            'readerSource' : {
                'uri' : item.path,
                'cache' : true,
            },
        });
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
                        backgroundColor : 'blue',
                    }}> 
                <View 
                    style = {styles.buttonGroup}>
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
                            style={styles.buttonIcon} />
                    </TouchableOpacity>
                </View>
                <View
                    style = {styles.buttonGroup}>
                    <TouchableOpacity
                        onPress = {()=>this.onPressClose()}
                        style = {[
                            styles.button,
                            {   
                                height : 35, 
                                width : 32.5
                            }   
                        ]}>
                        <Icon 
                            name = 'md-close'
                            size = {40}
                            color = {'white'}
                            style={styles.buttonIcon} />
                    </TouchableOpacity>
                </View>
            </View>
        );  
    } 

    computeDetectedViewDimensions(){
        const setDimensions = (width, height)=>{
            console.log('success');
            //console.log(width);
            //console.log(height);
            const detectedViewDimensions = { 
                width : dimensions.width*0.3*height/width,
                height : dimensions.height*0.3,
                set : true,
            };  
            this.setState({
                'detectedViewDimensions' : detectedViewDimensions,
            }); 
        };  
        Image.getSize(
            this.state.preview,
            setDimensions,
            (err)=> console.log(err)
        );  
    } 
        
    onPressPDF(){
        const {PDF, Image} = this.state;
        this.setState({
            'PDF' : !PDF,
            'Image' : !Image,
        });
    }

    onPressImage(){
        const {PDF, Image} = this.state;
        this.setState({
            'PDF' : !PDF,
            'Image' : !Image,
        });
    }

    renderPreview(){
        return(
            <View 
                style = {{
                    flex : 3,
                    flexDirection : 'row',
                    alignItems : 'center',
                    justifyContent : 'center',
                    margin : 5,
                    padding : 5,
                    backgroundColor : 'red',
                }}>
                <View
                    style = {{
                        flex : 1,
                        justifyContents : 'center',
                        alignSelf : 'center',
                    }}>
                    <Image 
                        source = {{uri : this.state.preview}}
                        style = {{
                            alignSelf : 'center',
                            height : this.state.detectedViewDimensions.height,
                            width : this.state.detectedViewDimensions.width,
                        }}
                        resizeMode = {'contain'}/>
                </View>
                <View style = {{flex : 1}}>
                    <View style = {{
                        //flex : 1,
                        backgroundColor : '#00000080',
                        opacity : 0.8,
                        height : 50,
                        margin : 5,
                        padding : 5,
                    }}>
                        <TextInput
                            onChangeText = {this.onChangeText}
                            placeholder = {'Add PDF name here'} 
                            style = {{
                                flex : 1,
                                margin : 2,
                                padding : 3, 
                                opacity : 1,
                                height : 50,
                                backgroundColor : 'white', }}
                            placeholderTextColor = {'black'}/>
                    </View>
                    <View 
                        style = {{
                            flex : 1,
                            flexDirection : 'row',
                            justifyContent : 'space-around',
                            alignItems : 'center',
                            padding : 5,
                            margin : 5,
                        }}>
                        <View 
                            style = {{
                                flex : 1,
                                justifyContent : 'center',
                                alignItems : 'center',
                                padding : 5,
                                margin : 5,
                            }}>
                            <View
                                style = {[
                                    styles.buttonGroup,
                                    {backgroundColor : this.state.PDF?  '#008000' : '#00000080'}
                                ]}>
                                <TouchableOpacity
                                    onPress = {()=>this.onPressPDF()}
                                    style = {styles.button}>
                                    <Icon
                                        name = 'md-document'
                                        size = {40}
                                        color = {'white'}
                                        style={styles.buttonIcon} />
                                    <Text style={styles.buttonText}>PDF</Text>
                                </TouchableOpacity>
                            </View>
                        </View>
                        <View 
                            style = {{
                                flex : 1,
                                justifyContent : 'center',
                                alignItems : 'center',
                                padding : 5,
                                margin : 5,
                            }}> 
                            <View
                                style = {[
                                    styles.buttonGroup,
                                    {backgroundColor : this.state.Image?'#008000' : '#00000080'}
                                ]}>
                                <TouchableOpacity
                                    onPress = {()=>this.onPressImage()}
                                    style = {styles.button}>
                                    <Icon
                                        name = 'md-image'
                                        size = {40}
                                        color = {'white'}
                                        style={styles.buttonIcon} />
                                    <Text style={styles.buttonText}>Image</Text>
                                </TouchableOpacity>
                            </View>
                        </View>
                    </View>
                    <View 
                        style = {{
                            flex : 1,
                            justifyContent : 'center',
                            alignItems : 'center',
                            padding : 5,
                            margin : 5
                        }}>
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
                                <Text style={styles.buttonText}>Save</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                </View>
            </View>
        );
    }
    
    renderLibrary(){
        return(
            <View style = {{flex : 6.5}}>
                <Library 
                    width = {'100%'}
                    numColumns = {2}
                    onPressPDFFile = {(item)=>this.onPressPDFFile(item)}/>
            </View>
        );
    }
    
    renderSaver(){
        //console.log(this.state);
        return(
            <View style = {[styles.container, {backgroundColor : 'white'}]}>
                {this.renderHeader()}
                {this.renderPreview()}
                {this.renderLibrary()}
            </View>
        );
    }

    renderReader(){
        return(
            <View>
                <Reader
                    source = {this.state.readerSource}/>
            </View>
        );
    }
  
    render(){
        if(this.state.mode == 'saver'){
            return this.renderSaver();
        }
        else if(this.state.mode == 'reader'){
            return this.renderReader();
        }
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
        flush : () => {dispatch(flushDoc())},
    };
};
//export default Saved;
export default connect(mapStateToProps, mapDispatchToProps)(Saved);
