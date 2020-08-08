import React, {Component} from 'react';
import {StyleSheet,
        TouchableHighlight,
        Dimensions,
        SafeAreaView,
        View,
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
                <View 
                    style={{flexDirection: 'row'}}>
                    <TouchableHighlight 
                        disabled={this.state.page === 1}
                        style={this.state.page === 1 ? styles.btnDisable : styles.btn}
                        onPress={() => this.prePage()}>
                        <Text 
                            style={styles.btnText}>{'-'}</Text>
                    </TouchableHighlight>
                    <View 
                        style={styles.btnText}>
                        <Text 
                            style={styles.btnText}>Page</Text>
                    </View>
                    <TouchableHighlight 
                        disabled={this.state.page === this.state.numberOfPages}
                        style={this.state.page === this.state.numberOfPages ? styles.btnDisable : styles.btn}
                        onPress={() => this.nextPage()}>
                        <Text 
                            style={styles.btnText}>{'+'}</Text>
                    </TouchableHighlight>
                    <TouchableHighlight 
                        disabled={this.state.scale === 1}
                        style={this.state.scale === 1 ? styles.btnDisable : styles.btn}
                        onPress={() => this.zoomOut()}>
                        <Text 
                            style={styles.btnText}>{'-'}</Text>
                    </TouchableHighlight>
                    <View 
                        style={styles.btnText}>
                        <Text 
                            style={styles.btnText}>Scale</Text>
                    </View>
                    <TouchableHighlight 
                        disabled={this.state.scale >= 3}
                        style={this.state.scale >= 3 ? styles.btnDisable : styles.btn}
                        onPress={() => this.zoomIn()}>
                        <Text 
                            style={styles.btnText}>{'+'}</Text>
                    </TouchableHighlight>
                    <View 
                        style={styles.btnText}>
                        <Text 
                            style={styles.btnText}>{'Horizontal:'}</Text>
                    </View>
                    <TouchableHighlight 
                        style={styles.btn} 
                        onPress={() => this.switchHorizontal()}>
                        {!this.state.horizontal ? (<Text 
                                                        style={styles.btnText}>{'false'}</Text>) : (
                            <Text 
                                style={styles.btnText}>{'true'}</Text>)}
                    </TouchableHighlight>
                </View>
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
            </SafeAreaView>
        )
    }
}

export default Reader;
