import React, {Component} from 'react';
import {View, Text, Dimensions} from 'react-native';
import {connect} from 'react-redux';
import RNImageToPdf from 'react-native-image-to-pdf';

const dimensions = Dimensions.get('window');

class Saved extends Component{
    constructor(props){
        super(props);
        console.log(props);
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
			    name: 'PDFName.pdf',
			    maxSize: { // optional maximum image dimension - larger images will be resized
				    width: 900,
				    height: Math.round(900*dimensions.height / dimensions.width),
			    },
			    quality: .7, // optional compression paramter
		    };
		    const pdf = await RNImageToPdf.createPDFbyImages(options);
		
		    console.log(pdf.filePath);
	    } catch(e) {
		    console.log(e);
	    }
    }
    
    componentDidMount(){
        this.convertToPDF();
    }

    render(){
        return(
            <View>
                <Text>Saved Screen</Text>
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
