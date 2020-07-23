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
const DocDir = RNFS.DocumentDirectoryPath;
console.log(DocDir);
const dimensions = Dimensions.get('window');
import {styles} from '../../assets/styles';
import Icon from 'react-native-vector-icons/Ionicons';

class Saved extends Component{
    constructor(props){
        super(props);
        //console.log(props);
        this.state = {
            fileName : 'MobScannerPDF.pdf'
        };
        this.onPressDone = this.onPressDone.bind(this);
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
		
		    console.log(pdf.filePath);
            const DestPath = `${DocDir}/${this.state.fileName}`;
            console.log(DestPath);
            const exists = await RNFS.exists(DestPath);
            if(exists){
                console.log('delete');
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
    
    render(){
        return(
            <View>
                <View
                    style = {{
                        flex : 1,
                        justifyContents : 'center',
                        alignItems : 'center',
                    }}>
                    <Image 
                        source = {{uri : this.props.doc[0].detectedImage}}
                        style = {{
                            height : 141,
                            width :  100
                        }}  
                        resizeMode = {'contain'}/>
                </View>
                <View>
                    <View>
                        <TextInput
                            onChangeText = {this.onChangeText}
                            placeholder = {'Add PDF name here'}/> 
                    </View>
                    <View style = {{}}>
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
//export default Saved;
export default connect(mapStateToProps, null)(Saved);
