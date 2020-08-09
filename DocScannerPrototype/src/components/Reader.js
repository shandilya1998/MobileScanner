import React, {Component} from 'react';
import {StyleSheet,
        TouchableHighlight,
        Dimensions,
        SafeAreaView,
        View,
        PanResponder,
        Animated,
        Text} from 'react-native';
import Pdf from 'react-native-pdf';
import Orientation from 'react-native-orientation-locker';
import {styles} from '../assets/styles';

class Reader extends Component{
    constructor(props) {
        super(props);
        this.state = {
            page: 1,
            scale: 1,
            numberOfPages: 0,
            horizontal: false,
            width: this.props.width
        };
        this.pdf = null;
        this.PanResponder = PanResponder.create({
            onStartShouldSetPanResponder: (evt, gestureState) => true,
            onPanResponderRelease: (evt, gestureState) => {
                let x = gestureState.dx;
                let y = gestureState.dy;
                if (Math.abs(x) > Math.abs(y)) {
                    if (x >= 0) {
                        this.onSwipePerformed('right')
                    }
                    else {
                        this.onSwipePerformed('left')
                    }
                }
                else {
                    if (y >= 0) {
                        this.onSwipePerformed('down')
                    }
                    else {
                        this.onSwipePerformed('up')
                    }
                }
            }
        })
    }

    onSwipePerformed(direction){
        if(direction=='up'){
           this.nextPage(); 
        }
        else if(direction == 'down'){
            this.prePage();
        }
    }

    _onOrientationDidChange = (orientation) => {
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

    render(){
        console.log(this.props.source);
        return (
            <SafeAreaView 
                style={[
                    styles.container,
                    {
                        justifyContent: 'flex-start',
                        alignItems: 'center',
                        marginTop: 25,
                        backgroundColor : 'white',
                    }
                ]}>
                <Animated.View
                    {...this.PanResponder.panHandlers}>
                    <View  
                        style={{flex:1,width: this.state.width}}>
                        <Pdf ref={(pdf) => {
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
                                    numberOfPages: numberOfPages 
                                }); 
                                console.log(`total page count: ${numberOfPages}`);
                                console.log(tableContents);
                            }}
                            onPageChanged={(page, numberOfPages) => {
                                this.setState({
                                    page: page
                                });
                                console.log(`current page: ${page}`);
                            }}
                            onError={(error) => {
                                console.log(error);
                            }}
                            style={{flex:1}}
                            />
                    </View>
                </Animated.View>
            </SafeAreaView>
        )
    }
}

export default Reader;
