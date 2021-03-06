import { PropTypes } from 'prop-types';
import React, { PureComponent } from 'react';
import {ActivityIndicator, 
        Animated, 
        Dimensions, 
        Platform, 
        SafeAreaView, 
        StatusBar, 
        StyleSheet, 
        Text, 
        TouchableOpacity, 
        View,
        Switch } from 'react-native';
import Icon from 'react-native-vector-icons/Ionicons';
import PDFScanner from '@woonivers/react-native-document-scanner';
import {styles} from '../../assets/styles';
import Permissions from 'react-native-permissions';
import {connect} from 'react-redux';
import {addPicture} from '../../actions/actions';

class Scanner extends PureComponent {
    constructor(props) {
        super(props);
        this.state = {
            flashEnabled: false,
            didLoadInitialLayout: false,
            isMultiTasking: false,
            loadingCamera: true,
            takingPicture: false,
            cameraAllowed : false,
            overlayFlashOpacity: new Animated.Value(0),
            captureMultiple : false,
            device: {
                initialized: false, 
                previewHeightPercent: 1,
                previewWidthPercent: 1,
            },
        };
        this.capture = this.capture.bind(this);
        this.toggleSwitch = this.toggleSwitch.bind(this);
        this.camera = React.createRef();
    }

    componentDidMount() {
        if (this.state.didLoadInitialLayout && !this.state.isMultiTasking) {
            this.turnOnCamera();
        }
    }

    componentDidUpdate() {
        if (this.state.didLoadInitialLayout) {
            if (this.state.isMultiTasking) return this.turnOffCamera(true);
            if (!this.state.showScannerView) {
                return this.turnOnCamera();
            }
        }
        return null;
    }

    componentWillUnmount() {
    }

    async requestCamera() {
        const result = await Permissions.request(
            Platform.OS === 'android'
            ? 'android.permission.CAMERA'
            : 'ios.permission.CAMERA',
        );  
        if (result === 'granted') {
            this.setState({cameraAllowed : true, loadingCamera : false});
        }   
    }
  
    getCameraDisabledMessage() {
        if (this.state.isMultiTasking) {
            return 'Camera is not allowed in multi tasking mode.';
        }
        const { cameraAllowed } = this.state;
    
        if (!cameraAllowed) {
            return 'Permission to use camera has not been granted.';
        }
        return 'Failed to set up the camera.';
    } 

    // On some android devices, the aspect ratio of the preview is different than
    // the screen size. This leads to distorted camera previews. This allows for correcting that.
    getPreviewSize() {
        const dimensions = Dimensions.get('window');
        // We use set margin amounts because for some reasons the percentage values don't align the camera preview in the center correctly.
        const heightMargin = (1 - this.state.device.previewHeightPercent) * dimensions.height / 2;
        const widthMargin = (1 - this.state.device.previewWidthPercent) * dimensions.width / 2;
        if (dimensions.height > dimensions.width) {
        // Portrait
            return {
                height: this.state.device.previewHeightPercent,
                width: this.state.device.previewWidthPercent,
                marginTop: heightMargin,
                marginLeft: widthMargin,
            };
        }

        // Landscape
        return {
            width: this.state.device.previewHeightPercent,
            height: this.state.device.previewWidthPercent,
            marginTop: widthMargin,
            marginLeft: heightMargin,
        };
    }

    capture(){
        if (this.state.takingPicture) return;
        this.setState({ takingPicture: true});
        this.camera.capture();
        this.triggerSnapAnimation();
    
        // If capture failed, allow for additional captures
     }

    // The picture was captured but still needs to be processed.
    onPictureTaken = (event) => {
        //console.log('test')
        this.setState({ takingPicture: false });
        this.props.onPictureTaken(event.initialImage, 
                                  event.croppedImage, 
                                  event.rectangleCoordinates);
        if(!this.state.captureMultiple){
            this.props.navigation.navigate('edit', 
                                          {'captureMultiple' : this.state.captureMultiple}); 
            this.turnOffCamera(); 
        }
    }

    // Flashes the screen on capture
    triggerSnapAnimation() {
        Animated.sequence([
            Animated.timing(this.state.overlayFlashOpacity, { toValue: 0.2, duration: 100 }),
            Animated.timing(this.state.overlayFlashOpacity, { toValue: 0, duration: 50 }),
            Animated.timing(this.state.overlayFlashOpacity, { toValue: 0.6, delay: 100, duration: 120 }),
            Animated.timing(this.state.overlayFlashOpacity, { toValue: 0, duration: 90 }),
        ]).start();
    } 

    // Hides the camera view. If the camera view was shown and onDeviceSetup was called,
    // but no camera was found, it will not uninitialize the camera state.
    turnOffCamera(shouldUninitializeCamera = false) {
        if (shouldUninitializeCamera && this.state.device.initialized) {
            this.setState(({ device }) => ({
                showScannerView: false,
                device: { ...device, initialized: false },
            }));
        } else if (this.state.showScannerView) {
            this.setState({ showScannerView: false });
        }
    }

    // Will show the camera view which will setup the camera and start it.
    // Expect the onDeviceSetup callback to be called
    turnOnCamera() {
        if (!this.state.showScannerView) {
            this.setState(({ device }) => ({
                showScannerView: true,
                loadingCamera : false,
                device: { ...device, initialized: false },
            }));
        }
    }

    // Renders the flashlight button. Only shown if the device has a flashlight.
    renderFlashControl() {
        const { flashEnabled, device } = this.state;
        return (
            <TouchableOpacity
                style={[styles.flashControl, { backgroundColor: flashEnabled ? '#FFFFFF80' : '#00000080' }]}
                activeOpacity={0.8}
                onPress={() => this.setState({ flashEnabled: !flashEnabled })}>
                <Icon 
                    name="ios-flashlight" 
                    style={
                        [styles.buttonIcon, 
                        { 
                            fontSize: 28, 
                            color: flashEnabled ? '#333' : '#FFF' }]} />
            </TouchableOpacity>
        );
    }

    toggleSwitch(){
        const value = this.state.captureMultiple;
        this.setState({captureMultiple : !value});
    }

    // Renders the camera controls. This will show controls on the side for large tablet screens
    // or on the bottom for phones. (For small tablets it will adjust the view a little bit).
    renderCameraControls() {
        const dimensions = Dimensions.get('window');
        const aspectRatio = dimensions.height / dimensions.width;
        const isPhone = aspectRatio > 1.6;
        const cameraIsDisabled = this.state.takingPicture;
        const disabledStyle = { opacity: cameraIsDisabled ? 0.8 : 1 };
        if (!isPhone) {
            if (dimensions.height < 500) {
            console.log('')
            return (
                <View style={styles.buttonContainer}>
                    <View style={[styles.buttonActionGroup, { flexDirection: 'row', alignItems: 'flex-end', marginBottom: 28 }]}>
                        {this.renderFlashControl()}
                        <View style={{ 
                            flexDirection: 'column', 
                            justifyContent: 'flex-end' }}>
                            {null}
                            <View style={{
                                backgroundColor: '#00000080',
                                flexDirection: isPhone ? 'column' : 'row',
                                borderRadius: 30,
                                margin: 8,}}>
                                <Switch
                                    trackColor={{ false: "#767577", true: "#81b0ff" }}
                                    thumbColor={this.state.captureMultiple ? "#f5dd4b" : "#f4f3f4"}
                                    ios_backgroundColor="#3e3e3e"
                                    onValueChange={this.toggleSwitch}
                                    value={isEnabled}
                                    style={{
                                        paddingHorizontal: 14,
                                        paddingVertical: 13,
                                        height: 50,
                                        width: 50,
                                    }}/>
                            </View>
                        </View>
                        <View style={[styles.buttonGroup, { marginLeft: 8 }]}>
                            <TouchableOpacity
                                style={[styles.button, disabledStyle]}
                                onPress={cameraIsDisabled ? () => null : this.props.onSkip}
                                activeOpacity={0.8}>
                                <Icon 
                                    name="md-arrow-round-forward" 
                                    size={40} 
                                    color="white" 
                                    style={styles.buttonIcon} />
                                <Text style={styles.buttonText}>Skip</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                    <View style={[styles.cameraOutline, disabledStyle]}>
                        <TouchableOpacity
                            activeOpacity={0.8}
                            style={styles.cameraButton}
                            onPress={this.capture}/>
                    </View>
                    <View style={
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
                                <Text style={styles.buttonText}>Cancel</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                </View>
            );
        }
        return (
            <View style={styles.buttonContainer}>
                <View style={[styles.buttonActionGroup, { justifyContent: 'flex-end', marginBottom: 20 }]}>
                    {this.renderFlashControl()}
                    <View style={{ 
                        flexDirection: 'column', 
                        justifyContent: 'flex-end' }}> 
                        {null}
                        <View style={{
                            backgroundColor: '#00000080',
                            flexDirection: isPhone ? 'column' : 'row',
                            borderRadius: 30, 
                            margin: 8,}}>
                            <Switch
                                trackColor={{ false: "#767577", true: "#81b0ff" }}
                                thumbColor={this.state.captureMultiple ? "#f5dd4b" : "#f4f3f4"}
                                ios_backgroundColor="#3e3e3e"
                                onValueChange={this.toggleSwitch}
                                value={this.state.captureMultiple}
                                style={{
                                    paddingHorizontal: 14, 
                                    paddingVertical: 13, 
                                    height: 50, 
                                    width: 50, }}/>
                        </View>
                    </View>
                </View>
                <View style={[styles.cameraOutline, disabledStyle]}>
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
                            <Text style={styles.buttonText}>Skip</Text>
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
                            <Text style={styles.buttonText}>Cancel</Text>
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
                        <Text style={styles.buttonText}>Cancel</Text>
                    </TouchableOpacity>
                </View>
                <View style={[styles.cameraOutline, disabledStyle]}>
                    <TouchableOpacity
                        activeOpacity={0.8}
                        style={styles.cameraButton}
                        onPress={this.capture}/>
                </View>
                <View>
                    <View style={
                        [styles.buttonActionGroup, 
                        { 
                            justifyContent: 'flex-end', 
                            marginBottom: this.props.hideSkip ? 0 : 16 }]}>
                        {this.renderFlashControl()}
                        <View style={{
                            flexDirection: 'column',
                            justifyContent: 'flex-end' }}>
                            <View style={{
                                backgroundColor: '#00000080',
                                flexDirection: isPhone ? 'column' : 'row',
                                borderRadius: 30,
                                margin: 8,}}>
                                <Switch
                                    trackColor={{ false: "#767577", true: "#81b0ff" }}
                                    thumbColor={this.state.captureMultiple ? "#f5dd4b" : "#f4f3f4"}
                                    ios_backgroundColor="#3e3e3e"
                                    onValueChange={this.toggleSwitch}
                                    value={this.state.captureMultiple}
                                    style={{
                                        paddingHorizontal: 14,
                                        paddingVertical: 13,
                                        height: 50,
                                        width: 50, }}/>
                            </View>
                        </View>
                    </View>
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
                            <Text style={styles.buttonText}>Skip</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </View>
        </>);
    }

    // Renders the camera controls or a loading/processing state
    renderCameraOverlay() {
        let loadingState = null;
        if (this.state.loadingCamera) {
            loadingState = (
                <View style={styles.overlay}>
                    <View style={styles.loadingContainer}>
                        <ActivityIndicator color="white" />
                        <Text 
                        style={styles.loadingCameraMessage}>Loading Camera</Text>
                    </View>
                </View>
            );
        }
        return (
            <>
                {loadingState}
                <SafeAreaView style={[styles.overlay]}>
                    {this.renderCameraControls()}
                </SafeAreaView>
            </>
        );
    }

    // Renders either the camera view, a loading state, or an error message
    // letting the user know why camera use is not allowed
    renderCameraView() {
        if (this.state.showScannerView) {
            const dimensions = Dimensions.get('window');
            const previewSize = this.getPreviewSize();
            let rectangleOverlay = null;
            // NOTE: I set the background color on here because for some reason the view doesn't line up correctly otherwise. It's a weird quirk I noticed.
            return (
            <View style={{ 
                backgroundColor: 'rgba(0, 0, 0, 0)', 
                position: 'relative', 
                marginTop: previewSize.marginTop, 
                marginLeft: previewSize.marginLeft, 
                height: dimensions.height, 
                width: dimensions.width }}>
                <PDFScanner
                    onDeviceSetup = {(event)=>this.onDeviceSetup(event)}
                    onPictureTaken={this.onPictureTaken}
                    overlayColor="rgba(135, 130, 235, 0.7)"
                    enableTorch={this.state.flashEnabled}
                    ref={ref =>{this.camera = ref;}}
                    useBase64 = {true}
                    captureMultiple = {this.state.captureMultiple}
                    quality={0.6}
                    manualOnly = {true}
                    style={styles.scanner}
                    detectionCountBeforeCapture={5000000}
                    detectionRefreshRateInMs={50}/>
                {rectangleOverlay}
                <Animated.View 
                    style={{
                        ...styles.overlay, 
                        backgroundColor: 'white', 
                        opacity: this.state.overlayFlashOpacity }} />
                {this.renderCameraOverlay()}
            </View>
        );
    }

    let message = null;
        if (this.state.loadingCamera) {
            message = (
                <View style={styles.overlay}>
                    <View style={styles.loadingContainer}>
                        <ActivityIndicator color="white" />
                        <Text 
                            style={styles.loadingCameraMessage}>Loading Camera</Text>
                    </View>
                </View>
            );
        } else {
             message = (
                <Text style={styles.cameraNotAvailableText}>
                    {this.getCameraDisabledMessage()}
                </Text>);
            }

        return (
            <View style={styles.cameraNotAvailableContainer}>
                {message}
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
                            <Text style={styles.buttonText}>Cancel</Text>
                        </TouchableOpacity>
                    </View>
                    <View style={styles.buttonGroup}>
                        <TouchableOpacity
                            style={[styles.button, { marginTop: 8 }]}
                            onPress={this.props.onSkip}
                            activeOpacity={0.8}>
                            <Icon 
                                name="md-arrow-round-forward" 
                                size={40} 
                                color="white" 
                                style={styles.buttonIcon} />
                            <Text style={styles.buttonText}>Skip</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </View>
        );
    }
    
    onDeviceSetup(event){
        console.log(event);
    }

    render() {
        return (
            <View
                style={styles.container}
                onLayout={(event) => {
                // This is used to detect multi tasking mode on iOS/iPad
                // Camera use is not allowed
                this.requestCamera();
                if (this.state.didLoadInitialLayout && Platform.OS === 'ios') {
                    const screenWidth = Dimensions.get('screen').width;
                    const isMultiTasking = (
                        Math.round(event.nativeEvent.layout.width) < Math.round(screenWidth));
                    if (isMultiTasking) {
                        this.setState({ isMultiTasking: true, loadingCamera: false });
                    } else {
                        this.setState({ isMultiTasking: false });
                    }
                } else {
                    this.setState({ didLoadInitialLayout: true });
                    }
                }}>
                <StatusBar 
                    backgroundColor="black" 
                    barStyle="light-content" 
                    hidden={Platform.OS !== 'android'} />
                    {this.renderCameraView()}
            </View>
        );
    }
}

const mapStateToProps = (state) => {
    //const {} = state;
    return {};
};

const mapDispatchToProps = (dispatch) => {
    return {
        onPictureTaken : (originalImage, detectedImage, rectCoords) => dispatch(addPicture(originalImage, detectedImage, rectCoords)), 
    };
};

export default connect(mapStateToProps, mapDispatchToProps)(Scanner);
