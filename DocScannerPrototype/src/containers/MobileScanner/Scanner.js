import React, {Component} from 'react';
import {
    View, 
    Text, 
    Animated, 
    SafeAreaView, 
    Dimensions, 
    Platform, 
    TouchableOpacity,
    } from 'react-native';
import PDFScanner from '@woonivers/react-native-document-scanner';
import {styles} from '../../assets/styles';
 
class Scanner extends Component{
    constructor(props){
        super(props);
        this.camera = React.createRef();
        this.state = {
            takingPicture : false,        
            flashEnabled : false,
            didLoadInitialLayout : false,
            inMultiTasking : false,
        };
        this.onPictureTaken = this.onPictureTaken.bind(this);
        this.capture = this.capture.bind(this);
    }
    
    triggerSnapAnimation() {
        Animated.sequence([
        Animated.timing(this.state.overlayFlashOpacity, { toValue: 0.2, duration: 100 }), 
        Animated.timing(this.state.overlayFlashOpacity, { toValue: 0, duration: 50 }),
        Animated.timing(this.state.overlayFlashOpacity, { toValue: 0.6, delay: 100, duration: 120 }), 
        Animated.timing(this.state.overlayFlashOpacity, { toValue: 0, duration: 90 }),
        ]).start();
    }
        
    capture(){
        if (this.state.takingPicture) return;
        this.setState({ takingPicture: true});
        this.camera.capture();
        this.triggerSnapAnimation();
    }

    onPictureTaken(event){
        this.setState({takingPicture : false});
        this.props.onPictureTaken
    }
        
    renderFlashControl() {
        const { flashEnabled, device } = this.state;
        if (!device.flashIsAvailable) return null;
        return (
            <TouchableOpacity
                style={
                    [styles.flashControl, 
                    {backgroundColor : flashEnabled ? '#FFFFFF80' : '#00000080' }]}
                activeOpacity={0.8}
                onPress={() => this.setState({ flashEnabled: !flashEnabled })}
      >
                    <Icon 
                        name="ios-flashlight" 
                        style={
                            [styles.buttonIcon, 
                            {
                                fontSize: 28, 
                                color: flashEnabled ? '#333' : '#FFF' 
                            }
                            ]}/>
            </TouchableOpacity>
        );
    }

    renderCameraControls() {
        const dimensions = Dimensions.get('window');
        const aspectRatio = dimensions.height / dimensions.width;
        const isPhone = aspectRatio > 1.6;
        const cameraIsDisabled = this.state.takingPicture;
        const disabledStyle = { opacity: cameraIsDisabled ? 0.8 : 1 };
        if (!isPhone) {
            if (dimensions.height < 500) {
                return (
                    <View style={styles.buttonContainer}>
                        <View 
                            style={
                                [styles.buttonActionGroup, 
                                { 
                                    flexDirection: 'row', 
                                    alignItems: 'flex-end', 
                                    marginBottom: 28 
                                }]}>
                                {this.renderFlashControl()}
                            <View 
                                style={
                                    [styles.buttonGroup, 
                                    { marginLeft: 8 }]}>
                                <TouchableOpacity
                                    style={
                                        [styles.button, 
                                        disabledStyle]}
                                        onPress={cameraIsDisabled ? () => null : this.props.onSkip}
                                        activeOpacity={0.8}>
                                    <Icon 
                                        name="md-arrow-round-forward" 
                                        size={40} 
                                        color="white" 
                                        style={styles.buttonIcon} />
                                    <Text 
                                        style={styles.buttonText}>
                                        Skip
                                    </Text>
                                </TouchableOpacity>
                            </View>
                        </View>
                        <View
                            style={
                                [styles.cameraOutline, 
                                disabledStyle]}>
                            <TouchableOpacity
                                activeOpacity={0.8}
                                style={styles.cameraButton}
                                onPress={this.capture}/>
                        </View>
                        <View 
                            style={
                                [styles.buttonActionGroup, 
                                { marginTop: 28 }]}>
                            <View style={styles.buttonGroup}>
                                <TouchableOpacity
                                    style={styles.button}
                                    onPress={this.props.onCancel}
                                    activeOpacity={0.8}>
                                    <Icon 
                                        name="ios-close-circle" 
                                        size={40} 
                                        style={styles.buttonIcon} />
                                    <Text 
                                        style={styles.buttonText}>
                                        Cancel
                                    </Text>
                                </TouchableOpacity>
                            </View>
                        </View>
                    </View>
                );
            }
            return (
                <View style={styles.buttonContainer}>
                    <View style={
                        [styles.buttonActionGroup, 
                        { justifyContent: 'flex-end', marginBottom: 20 }]}>
                        {this.renderFlashControl()}
                    </View>
                    <View 
                        style={
                            [styles.cameraOutline, 
                            disabledStyle]}>
                        <TouchableOpacity
                            activeOpacity={0.8}
                            style={styles.cameraButton}
                            onPress={this.capture}/>
                    </View>
                    <View 
                        style={
                            [styles.buttonActionGroup, 
                            { marginTop: 28 }]}>
                    <View style={styles.buttonGroup}>
                        <TouchableOpacity
                            style={[styles.button, disabledStyle]}
                            onPress={cameraIsDisabled ? () => null : this.props.onSkip}
                            activeOpacity={0.8}>
                            <Icon 
                                name="md-arrow-round-forward" 
                                size={40} 
                                color="white" 
                                style={styles.buttonIcon} />
                            <Text 
                                style={styles.buttonText}>
                                Skip
                            </Text>
                        </TouchableOpacity>
                    </View>
                    <View style={styles.buttonGroup}>
                        <TouchableOpacity
                            style={styles.button}
                            onPress={this.props.onCancel}
                            activeOpacity={0.8}>
                            <Icon 
                                name="ios-close-circle" 
                                size={40} 
                                style={styles.buttonIcon} />
                            <Text 
                                style={styles.buttonText}>
                                Cancel
                            </Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </View>
        );
    }
    return (
        <>
            <View style={styles.buttonBottomContainer}>
                <View style={styles.buttonGroup}>
                    <TouchableOpacity
                        style={styles.button}
                        onPress={this.props.onCancel}
                        activeOpacity={0.8}>
                        <Icon 
                            name="ios-close-circle" 
                            size={40} 
                            style={styles.buttonIcon} />
                        <Text 
                            style={styles.buttonText}>
                            Cancel
                        </Text>
                    </TouchableOpacity>
                </View>
                <View 
                    style={[styles.cameraOutline, disabledStyle]}>
                    <TouchableOpacity
                        activeOpacity={0.8}
                        style={styles.cameraButton}
                        onPress={this.capture}/>
                </View>
                <View>
                    <View 
                        style={
                            [styles.buttonActionGroup, 
                            {
                                justifyContent: 'flex-end', 
                                marginBottom: this.props.hideSkip ? 0 : 16 
                            }]}>
                        {this.renderFlashControl()}
                    </View>
                    <View style={styles.buttonGroup}>
                        <TouchableOpacity
                            style={[
                                styles.button, 
                                disabledStyle]}
                            onPress={cameraIsDisabled ? () => null : this.props.onSkip}
                            activeOpacity={0.8}>
                            <Icon 
                                name="md-arrow-round-forward" 
                                size={40} 
                                color="white" 
                                style={styles.buttonIcon} />
                            <Text 
                                style={styles.buttonText}>
                                Skip
                            </Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </View>
        </>);
    }
    
    renderCameraOverlay(){
        return(
            <>
                <SafeAreaView style={[styles.overlay]}>
                    {this.renderCameraControls()}
                </SafeAreaView>
            </>
        );
    }   

    renderCameraView(){
        return(
            <View
                style={{ 
                    backgroundColor: 'rgba(0, 0, 0, 0)', 
                    position: 'relative', 
                    marginTop: 5, 
                    marginLeft: 5, 
                    flex : 1}}>
                <PDFScanner
                    onPictureTaken={this.onPictureTaken}
                    overlayColor="rgba(135, 130, 235, 0.7)"
                    enableTorch={this.state.flashEnabled}
                    ref={this.camera}
                    quality={0.6}
                    manualOnly = {false}
                    style={styles.scanner}
                    detectionCountBeforeCapture={5000000}
                    detectionRefreshRateInMs={50}/>
                <Animated.View style={{ ...styles.overlay, backgroundColor: 'white', opacity: this.state.overlayFlashOpacity }} />
                {this.renderCameraOverlay()}
            </View>
        );
    }
    
    render(){
        return(
            <View
                style = {[styles.container, {backgroundColor:'white'}]}>
                onLayout={(event) => {
                    // This is used to detect multi tasking mode on iOS/iPad
                    // Camera use is not allowed
                    if (this.state.didLoadInitialLayout && Platform.OS === 'ios') {
                        const screenWidth = Dimensions.get('screen').width;
                        const isMultiTasking = (
                            Math.round(event.nativeEvent.layout.width) < Math.round(screenWidth));
                        if (isMultiTasking) {
                            this.setState({ 
                                isMultiTasking: true, 
                                loadingCamera: false 
                            });
                        } else {
                            this.setState({ isMultiTasking: false });
                        }
                    } else {
                        this.setState({ didLoadInitialLayout: true });
                    }}}>
                {this.renderCameraView()}
            </View>
        ); 
    }
}

export default Scanner;
