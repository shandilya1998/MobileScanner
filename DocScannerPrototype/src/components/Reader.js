import React, {Component} from 'react';
import {StyleSheet,
        TouchableOpacity,
        Dimensions,
        ActivityIndicator,
        SafeAreaView,
        View,
        PanResponder,
        Animated,
        Text} from 'react-native';
import Pdf from 'react-native-pdf';
import Orientation from 'react-native-orientation-locker';
import {styles} from '../assets/styles';
import Icon from 'react-native-vector-icons/Ionicons';

class Reader extends Component{
    constructor(props) {
        super(props);
        this.state = {
            page: 1,
            scale: 1,
            numberOfPages: 0,
            horizontal: false,
            loading : true,
            pdfWidth : this.props.width,
            pdfHeight : this.props.height,
        };
        this.pdf = null;
        this.PanResponder = PanResponder.create({
            onStartShouldSetPanResponder: (evt, gestureState) => true,
            onPanResponderRelease: (evt, gestureState) => {
                let x = gestureState.dx;
                let y = gestureState.dy;
                if (Math.abs(x) > Math.abs(y)) {
                    if (x > 0) {
                        this.onSwipePerformed('right')
                    }
                    else if(x<0)  {
                        this.onSwipePerformed('left')
                    }
                }
                else {
                    if (y > 0) {
                        this.onSwipePerformed('down')
                    }
                    else if(y<0) {
                        this.onSwipePerformed('up')
                    }
                }
            }
        });
        this.renderPdf = this.renderPdf.bind(this);
        this.onSwipePerformed = this.onSwipePerformed.bind(this);
        this._onOrientationDidChange = this._onOrientationDidChange.bind(this);
    }

    onSwipePerformed(direction){
        if(direction=='up'){
           this.nextPage(); 
        }
        else if(direction == 'down'){
            this.prePage();
        }
    }

    _onOrientationDidChange(orientation){
        if (orientation == 'LANDSCAPE-LEFT'||orientation == 'LANDSCAPE-RIGHT') {
          this.setState({width:this.props.height>this.props.width?this.props.height:this.props.width,horizontal:true});
        } else {
          this.setState({width:this.props.height>this.props.width?this.props.height:this.props.width,horizontal:false});
        }
    };

    componentDidMount() {
        Orientation.addOrientationListener(this._onOrientationDidChange);
    }

    componentWillUnmount() {
        Orientation.removeOrientationListener(this._onOrientationDidChange);
    }

    prePage = () => {
        let prePage = this.state.page > 1 ? this.state.page - 1 : 1;
        this.pdf.setPage(prePage);
        console.log(`prePage: ${prePage}`);
    };

    nextPage = () => {
        let nextPage = this.state.page + 1 > this.state.numberOfPages ? this.state.numberOfPages : this.state.page + 1;
        this.pdf.setPage(nextPage);
        console.log(`nextPage: ${nextPage}`);
    };

    zoomOut = () => {
        let scale = this.state.scale > 1 ? this.state.scale / 1.2 : 1;
        this.setState({scale: scale});
        console.log(`zoomOut scale: ${scale}`);
    };

    zoomIn = () => {
        let scale = this.state.scale * 1.2;
        scale = scale > 3 ? 3 : scale;
        this.setState({scale: scale});
        console.log(`zoomIn scale: ${scale}`);
    };

    switchHorizontal = () => {
        this.setState({horizontal: !this.state.horizontal, page: this.state.page});
    };
    
    renderPdf(){
        return(
            <Pdf 
                ref={(pdf) => {
                    this.pdf = pdf;
                }}  
                source={this.props.source}
                scale={this.state.scale}
                horizontal={this.state.horizontal}
                onLoadComplete={(
                    numberOfPages, 
                    filePath,
                    {   
                        width,
                        height
                    },  
                    tableContents
                ) => {
                    this.setState({
                        numberOfPages: numberOfPages,
                    }); 
                }}  
                onPageChanged={(page, numberOfPages) => {
                    this.setState({
                        page: page
                    }); 
                }}  
                onError={(error) => {
                    console.log(error);
                }}  
                enablePaging = {true}
                fitPolicy = {1} 
                style={{
                    flex : 1,
                }}/>
        ); 
    }

    render(){
        const loading = () => {
            return (
                <View
                    style={[
                        styles.overlay,
                        {   
                            backgroundColor : 'black',
                            //width : Dimensions.get('window').width, 
                        }]}>
                    <View style={styles.loadingContainer}>
                        <ActivityIndicator color="white" />
                        <Text 
                            style={styles.loadingCameraMessage}>
                            Loading Reader
                        </Text>
                    </View>
                </View>
            );
        };
        return (
            <SafeAreaView 
                style={[
                    styles.container,
                    {
                        justifyContent: 'flex-start',
                        alignItems: 'center',
                        backgroundColor : 'white',
                    }
                ]}>
                <View 
                    style={
                        {
                            flex : 0.5,
                            width : this.props.width,
                            flexDirection: 'row',
                            justifyContent : 'space-between',
                            marginHorizontal : 5,
                            padding : 5,
                            alignItems : 'center',
                        }
                    }>
                    <View 
                        style = {styles.buttonGroup}>
                        <TouchableOpacity
                            style = {[
                                styles.button,
                                {   
                                    height : 35, 
                                    width : 32.5
                                }   
                            ]}
                            onPress = {()=>{this.props.onPressMore()}}>
                            <Icon 
                                name = 'md-more'
                                size = {40}
                                color = {'white'}
                                style={styles.buttonIcon} /> 
                        </TouchableOpacity>
                    </View>
                    <View 
                        style={styles.btnText}>
                        <Text 
                            style={styles.btnText}>
                            Page {this.state.page}
                        </Text>
                    </View>
                    <View 
                        style = {styles.buttonGroup}>
                        <TouchableOpacity
                            style = {[
                                styles.button,
                                {   
                                    height : 35, 
                                    width : 32.5
                                }   
                            ]}
                            onPress = {()=>{this.props.onPressClose()}}> 
                            <Icon 
                                name = 'md-close'
                                size = {40}
                                color = {'white'}
                                style={styles.buttonIcon} />
                        </TouchableOpacity>
                    </View>
                </View>
                <Animated.View
                    {...this.PanResponder.panHandlers}
                    style = {{
                        flex : 9,
                    }}>
                    <View
                        onLayout = {
                            (event) => {
                                this.setState({
                                    loading : false,
                                    pdfHeight : event.nativeEvent.layout.height,
                                    pdfWidth : event.nativeEvent.layout.width,
                                })}
                        }
                        style = {{
                            width : this.state.pdfWidth,
                            height : this.state.pdfHeight,
                        }}> 
                        {this.renderPdf()} 
                    </View>
                </Animated.View>
                {this.state.loading?loading():null}
            </SafeAreaView>
        )
    }
}

export default Reader;
